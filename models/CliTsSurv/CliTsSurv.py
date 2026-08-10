import numpy as np
import torch
from torch import nn
import os
from .backbones.LatentEncoder import LatentEncoder
from .backbones.CliTs_backbone import CliTsSurv_backbone
from .backbones.time_embedding import build_time_encoder
from .metrices import deephit_loss, brier_loss
from sksurv.metrics import concordance_index_censored
from .augmenters.seq_augmenters import *


class CliTsSurv(nn.Module):
    """DeepHit version for survival analysis with static features and augmentation"""
    def __init__(self, conti_feature_in, bi_feature_in, static_conti_feature_in, static_bi_feature_in,
                 time_dim, encoder_args, static_encoder_args, net_args, head_args,
                 beta1=1, beta2=1, aug_configs=[], num_time_bins=10, alpha=0.1, sigma=0.1, gamma=0.0,
                 time_encoder_type='sincos', device='cuda'):
        super(CliTsSurv, self).__init__()

        self.encoder = LatentEncoder(conti_feature_in, bi_feature_in, 
                                     enc_dims=encoder_args['enc_dims'], 
                                     dec_dims=encoder_args['dec_dims'],
                                     disc_dims=encoder_args.get('disc_dims', []),
                                     beta=encoder_args.get('beta', 1))
        self.static_encoder = LatentEncoder(static_conti_feature_in, static_bi_feature_in, 
                                            enc_dims=static_encoder_args['enc_dims'],
                                            dec_dims=static_encoder_args['dec_dims'],
                                            disc_dims=static_encoder_args.get('disc_dims', []),
                                            beta=static_encoder_args.get('beta', 1))
        self.hidden_dim = self.encoder.neck_dim
        self.net = CliTsSurv_backbone(self.hidden_dim + time_dim, **net_args)
        self.head = CliTsSurv_head(self.net.hidden_size + self.encoder.neck_dim + time_dim + self.static_encoder.neck_dim, 
                               num_time_bins=num_time_bins, **head_args)
        
        self.device = device        
        self.alpha = alpha
        self.sigma = sigma
        self.num_time_bins = num_time_bins
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        
        self.augmenters = []
        for aug in aug_configs:
            self.augmenters.append(eval(aug['type'])(device=self.device, **aug['args']))

        # Time encoder: 'sincos' (fixed) | 'time2vec' | 'ctlpe' (learnable)
        self.time_encoder = build_time_encoder(time_encoder_type, time_dim).to(device)

        # Store time_bin_edges for prediction (will be set during training)
        self.time_bin_edges = None


    def loss_fn(self, pred, time_bins, event_indicator,
                recon, recon_target, static_recon, static_recon_target,
                recon_mask=None, disc_pred=None, disc_label=None,
                static_recon_mask=None, static_disc_pred=None, static_disc_label=None):
        """Combined DeepHit + VAE reconstruction + Brier calibration loss"""
        loss_enc, _, _ = self.encoder.loss_fn(recon, recon_target, recon_mask, disc_pred, disc_label)
        loss_static_enc, _, _ = self.static_encoder.loss_fn(static_recon, static_recon_target,
                                                             static_recon_mask, static_disc_pred,
                                                             static_disc_label)
        loss_survival = deephit_loss(pred, time_bins, event_indicator, self.alpha, self.sigma)
        loss_brier = brier_loss(pred, time_bins, event_indicator, self.num_time_bins)

        loss_enc = torch.where(torch.isnan(loss_enc), 0, loss_enc)
        loss_static_enc = torch.where(torch.isnan(loss_static_enc), 0, loss_static_enc)

        total_loss = (loss_survival
                      + self.beta1 * loss_enc
                      + self.beta2 * loss_static_enc
                      + self.gamma * loss_brier)
        return total_loss, loss_survival, loss_enc, loss_static_enc, loss_brier
            
        
    def build_mask(self, lens, T, K):
        masks = []
        for len in lens:
            ones = np.ones((len, K))
            zeros = np.zeros((T-len, K))
            masks.append(torch.unsqueeze(torch.from_numpy(np.vstack([ones, zeros]).astype(np.float32)), dim=0))
        mask = torch.concatenate(masks).to(self.device)
        return mask
    
    
    def augment(self, z, lens, times, labels=None):
        for augmenter in self.augmenters:
            z, lens, times = augmenter(z, lens, times, labels)
        return z, lens, times
    

    def forward(self, x, static_x, lens, times, labels=None,
                return_attention=False, return_latents=False):
        B, T, K = x.shape

        z, recon, disc_pred = self.encoder(x.view((B*T, K)))
        static_z, static_recon, static_disc_pred = self.static_encoder(static_x)

        mask_recon = self.build_mask(lens, T, K)
        recon = recon.view((B, T, K))
        recon = recon * mask_recon

        mask = self.build_mask(lens, T, self.hidden_dim)
        z = z.view((B, T, self.hidden_dim))
        z = z * mask

        # augmentation
        if self.training and len(self.augmenters) > 0:
            z, lens, times = self.augment(z, lens, times, labels)

        # time encoding: times is raw (B, T, 1), encode to (B, T, time_dim)
        time_emb = self.time_encoder(times)
        z_in = torch.concat([z, time_emb], dim=-1)

        backbone_attention = None
        if return_attention:
            z_agg, backbone_attention = self.net(z_in, lens, return_attention=True)
        else:
            z_agg = self.net(z_in, lens)

        z_fused = torch.concat([z_agg, static_z], dim=-1)
        out = self.head(z_fused)

        outputs = (out, recon, disc_pred, static_recon, static_disc_pred)

        extras = {}
        if return_attention:
            extras['attention'] = backbone_attention
        if return_latents:
            extras['latents'] = {'z': z, 'static_z': static_z, 'times': times}
        if extras:
            return outputs + (extras,)
        return outputs

    def encode(self, x, static_x, lens):
        """Encode raw inputs into latent representations without survival head.

        Args:
            x:        (B, T, K) temporal features
            static_x: (B, S) static features
            lens:     (B,) actual sequence lengths

        Returns:
            z:        (B, T, hidden_dim) masked temporal latent sequence
            static_z: (B, static_neck_dim) static latent vector
            z_last:   (B, hidden_dim) latent at each sample's last valid visit
        """
        self.eval()
        B, T, K = x.shape
        with torch.no_grad():
            z, _, _ = self.encoder(x.view((B * T, K)))
            static_z, _, _ = self.static_encoder(static_x)

            mask = self.build_mask(lens, T, self.hidden_dim)
            z = z.view((B, T, self.hidden_dim)) * mask

            if not isinstance(lens, torch.Tensor):
                lens_t = torch.tensor(lens, dtype=torch.long, device=z.device)
            else:
                lens_t = lens.to(z.device)
            last_idx = (lens_t - 1).clamp(min=0)
            idx = last_idx.view(B, 1, 1).expand(B, 1, z.size(2))
            z_last = z.gather(1, idx).squeeze(1)

        return z, static_z, z_last


    def train_model(self, train_dataloader, test_dataloader, epochs=100, lr=0.0001,
                    save_dir='./train', time_bin_edges=None,
                    optimizer_type='adam', optimizer_kwargs=None,
                    early_stopping_patience=8):
        """
        Args:
            time_bin_edges: array of time bin boundaries for converting continuous time to bins
            optimizer_type: optimizer to use, one of 'adam' | 'adamw' | 'sgd' (default: 'adam')
            optimizer_kwargs: extra keyword arguments forwarded to the optimizer constructor
                              (e.g. {'weight_decay': 1e-4} for AdamW). lr is always set separately.
            early_stopping_patience: stop training if score does not improve for this many epochs.
                                     Set to None or 0 to disable.
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Save time_bin_edges for prediction
        if time_bin_edges is not None:
            self.time_bin_edges = time_bin_edges

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        param_groups = [
            {'params': self.encoder.enc.parameters()},
            {'params': self.encoder.dec.parameters()},
            {'params': self.static_encoder.enc.parameters()},
            {'params': self.static_encoder.dec.parameters()},
            {'params': self.net.parameters()},
            {'params': self.head.parameters()},
        ]

        _opt_type = optimizer_type.lower()
        if _opt_type == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, lr=lr, **optimizer_kwargs)
        elif _opt_type == 'sgd':
            optimizer = torch.optim.SGD(param_groups, lr=lr, **optimizer_kwargs)
        else:  # default: adam
            optimizer = torch.optim.Adam(param_groups, lr=lr, **optimizer_kwargs)

        best_score = 0
        best_epoch = 0
        loss_hist = []
        global_step = 0
        no_improve_count = 0
        
        for epoch in range(epochs):
            self.train()
            epoch_losses = []
            epoch_survival_losses = []
            epoch_rec_losses = []
            epoch_static_rec_losses = []
            epoch_brier_losses = []
            
            for step, batch in enumerate(train_dataloader):
                # Expected: (x, static_x, time_to_event, event_indicator, lens, times, mask_nan, mask_nan_static)
                x = batch[0]
                static_x = batch[1]
                time_to_event = batch[2]
                event_indicator = batch[3]
                lens = batch[4]
                times = batch[5]
                mask_nan = batch[6]
                mask_nan_static = batch[7]
                
                pred, recon, disc_pred, static_recon, static_disc_pred = self.forward(
                    x, static_x, lens, times, labels=event_indicator
                )
                
                # Convert time to bin indices
                if time_bin_edges is not None:
                    time_bins = torch.searchsorted(
                        torch.tensor(time_bin_edges[1:], device=pred.device),
                        time_to_event
                    ).clamp(0, self.num_time_bins - 1)
                else:
                    time_bins = time_to_event.long()
                
                # Discriminator labels for VAE
                if disc_pred is not None:
                    disc_label = torch.ones(disc_pred.shape, device=x.device)
                else:
                    disc_label = None
                if static_disc_pred is not None:
                    static_disc_label = torch.ones(static_disc_pred.shape, device=x.device)
                else:
                    static_disc_label = None
                    
                loss, loss_survival, loss_rec, loss_static_rec, loss_brier = self.loss_fn(
                    pred, time_bins, event_indicator,
                    recon, x, static_recon, static_x,
                    mask_nan, disc_pred, disc_label,
                    mask_nan_static, static_disc_pred, static_disc_label
                )
                
                optimizer.zero_grad()
                loss.backward()
                
                global_step += 1
                optimizer.step()

                epoch_losses.append(loss.detach().cpu().numpy())
                epoch_survival_losses.append(loss_survival.detach().cpu().numpy())
                epoch_rec_losses.append(loss_rec.detach().cpu().numpy())
                epoch_static_rec_losses.append(loss_static_rec.detach().cpu().numpy())
                epoch_brier_losses.append(loss_brier.detach().cpu().numpy())

            train_loss = np.mean(epoch_losses)
            train_survival_loss = np.mean(epoch_survival_losses)
            train_rec_loss = np.mean(epoch_rec_losses)
            train_static_rec_loss = np.mean(epoch_static_rec_losses)
            train_brier_loss = np.mean(epoch_brier_losses)
            loss_hist.append(str(train_loss))

            cindex, val_loss, val_survival_loss, val_brier_loss = self.test_model(test_dataloader, time_bin_edges)

            print('[Epoch %d] train loss: %.4f (survival: %.4f, rec: %.4f, static_rec: %.4f, brier: %.4f)' %
                  (epoch, train_loss, train_survival_loss, train_rec_loss, train_static_rec_loss, train_brier_loss))
            print('val loss: %.4f | val survival loss: %.4f | c-index: %.4f, brier: %.4f' % 
                  (val_loss, val_survival_loss, cindex, val_brier_loss))
            
            score = np.mean([cindex, 1 - val_brier_loss])  # You can also consider other metrics or a combination for model selection

            if score > best_score:
                torch.save(self.state_dict(), os.path.join(save_dir, './best.pt'))
                best_score = score
                best_epoch = epoch
                no_improve_count = 0
                print('update best score %.4f, model saved' % best_score)
            else:
                no_improve_count += 1

            if early_stopping_patience and no_improve_count >= early_stopping_patience:
                print(f'early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)')
                break

            if (epoch+1) % 10 == 0:
                torch.save(self.state_dict(), os.path.join(save_dir, 'epoch_%d.pt'%epoch))

        print('training finished', '='*50)
        print('best score %.4f at epoch %d' % (best_score, best_epoch))
        
        with open(os.path.join(save_dir, 'loss_hist'), 'w+', encoding='utf8', newline='') as f:
            loss_hist = list(map(lambda x: str(x)+'\n', loss_hist))
            f.writelines(loss_hist)

        self.load_state_dict(torch.load(os.path.join(save_dir, './best.pt'), weights_only=True))
        
        
    def test_model(self, test_dataloader, time_bin_edges=None):
        self.eval()

        all_pred_risks = []
        all_times = []
        all_events = []
        all_losses = []
        all_survival_losses = []
        all_brier_losses = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                x = batch[0]
                static_x = batch[1]
                time_to_event = batch[2]
                event_indicator = batch[3]
                lens = batch[4]
                times = batch[5]
                mask_nan = batch[6]
                mask_nan_static = batch[7]
                
                pred, recon, disc_pred, static_recon, static_disc_pred = self.forward(
                    x, static_x, lens, times, labels=event_indicator
                )
                
                # Convert time to bin indices
                if time_bin_edges is not None:
                    time_bins = torch.searchsorted(
                        torch.tensor(time_bin_edges[1:], device=pred.device),
                        time_to_event
                    ).clamp(0, self.num_time_bins - 1)
                else:
                    time_bins = time_to_event.long()
                
                # Discriminator labels for VAE
                if disc_pred is not None:
                    disc_label = torch.ones(disc_pred.shape, device=x.device)
                else:
                    disc_label = None
                if static_disc_pred is not None:
                    static_disc_label = torch.ones(static_disc_pred.shape, device=x.device)
                else:
                    static_disc_label = None
                    
                loss, loss_survival, _, _, loss_brier = self.loss_fn(
                    pred, time_bins, event_indicator,
                    recon, x, static_recon, static_x, 
                    mask_nan, disc_pred, disc_label,
                    mask_nan_static, static_disc_pred, static_disc_label
                )
                
                all_losses.append(loss.detach().cpu().numpy())
                all_survival_losses.append(loss_survival.detach().cpu().numpy())
                all_brier_losses.append(loss_brier.detach().cpu().numpy())
                
                # Risk score: cumulative event probability
                event_probs = pred[:, :-1]
                cumulative_event_prob = torch.cumsum(event_probs, dim=-1)
                risk_score = cumulative_event_prob[:, -1]
                
                all_pred_risks.extend(risk_score.cpu().numpy().tolist())
                all_times.extend(time_to_event.cpu().numpy().tolist())
                all_events.extend(event_indicator.cpu().numpy().tolist())
               
        cindex = concordance_index_censored(
            np.array(all_events, dtype=bool),
            np.array(all_times),
            np.array(all_pred_risks)
        )[0]
        
        return cindex, np.mean(all_losses), np.mean(all_survival_losses), np.mean(all_brier_losses)
    


    def _preprocess_test_data(self, x, static_x=None, lens=None, times=None):
        """Preprocess test data for prediction.
        
        Args:
            x: List of arrays or numpy array of sequences
            static_x: Static features (batch, static_dim) or None
            lens: Sequence lengths (batch,) or None (will compute from x)
            times: Time embeddings (batch, T, time_dim) or None
            
        Returns:
            x_tensor: (batch, T, feature_dim) tensor
            static_x_tensor: (batch, static_dim) tensor or None
            lens_tensor: (batch,) tensor
            times_tensor: (batch, T, time_dim) tensor or None
        """
        # Handle list of arrays (variable length sequences)
        if isinstance(x, list):
            if lens is None:
                lens = np.array([len(xi) for xi in x])
            max_len = max(lens)
            # Pad sequences to max length
            x_padded = []
            for xi in x:
                if len(xi) < max_len:
                    pad = np.zeros((max_len - len(xi), xi.shape[1]))
                    xi_padded = np.vstack([xi, pad])
                else:
                    xi_padded = xi
                x_padded.append(xi_padded)
            x = np.array(x_padded)
        else:
            if lens is None:
                # Assume all sequences are full length
                lens = np.array([x.shape[1]] * x.shape[0])
        
        x_tensor = torch.from_numpy(x).float().to(self.device)
        lens_tensor = torch.from_numpy(lens).long().to(self.device)
        
        # Handle static features
        if static_x is not None:
            if isinstance(static_x, list):
                static_x = np.array(static_x)
            static_x_tensor = torch.from_numpy(static_x).float().to(self.device)
        else:
            static_x_tensor = None
            
        # Handle time embeddings
        if times is not None:
            if isinstance(times, list):
                times_padded = []
                for i, ti in enumerate(times):
                    if len(ti) < max_len:
                        pad = np.zeros((max_len - len(ti), ti.shape[1] if len(ti.shape) > 1 else 1))
                        ti_padded = np.vstack([ti, pad]) if len(ti.shape) > 1 else np.concatenate([ti, pad.flatten()])
                    else:
                        ti_padded = ti
                    times_padded.append(ti_padded)
                times = np.array(times_padded)
            times_tensor = torch.from_numpy(times).float().to(self.device)
        else:
            # Create dummy time embeddings if not provided
            times_tensor = torch.zeros(x_tensor.shape[0], x_tensor.shape[1], 1).to(self.device)
            
        return x_tensor, static_x_tensor, lens_tensor, times_tensor

    def predict_survival(self, x, t, static_x=None, lens=None, times=None, risk=1, bs=100):
        """Predict survival function S(t) for given time points.
        
        Aligns with DDH's predict_survival behavior:
        - Converts continuous time t to bin indices using time_bin_edges
        - Returns survival probability S(t) = 1 - F(t) at specified time points
        
        Args:
            x: List of arrays or numpy array of sequences (batch, T, features)
            t: Time points at which to evaluate survival function (array or list)
            static_x: Static features (batch, static_features) or None
            lens: Sequence lengths (batch,) or None
            times: Time embeddings (batch, T, time_dim) or None
            risk: Risk index (default 1)
            bs: Batch size for prediction
            
        Returns:
            survival_probs: (batch, len(t)) array of survival probabilities at specified times
        """
        self.eval()
        
        # Handle input format
        if isinstance(x, list):
            n_samples = len(x)
            if lens is None:
                lens = np.array([len(xi) for xi in x])
        else:
            n_samples = x.shape[0]
            if lens is None:
                lens = np.array([x.shape[1]] * n_samples)
        
        # Ensure t is array
        if not isinstance(t, np.ndarray):
            t = np.array(t)
        
        # Convert continuous time to bin indices (same as DDH's discretize logic)
        if hasattr(self, 'time_bin_edges') and self.time_bin_edges is not None:
            # DDH uses np.digitize with right=True, which is equivalent to searchsorted with side='right'
            t_bins = np.digitize(t, self.time_bin_edges, right=True) - 1
            t_bins = np.clip(t_bins, 0, self.num_time_bins - 1)
        else:
            # Assume t is already bin indices
            t_bins = t.astype(int)
        
        # Batch prediction
        n_batches = int(np.ceil(n_samples / bs))
        all_survival_probs = []
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * bs
                end_idx = min((i + 1) * bs, n_samples)
                
                # Slice batch
                if isinstance(x, list):
                    x_batch = x[start_idx:end_idx]
                else:
                    x_batch = x[start_idx:end_idx]
                    
                lens_batch = lens[start_idx:end_idx]
                
                static_x_batch = None
                if static_x is not None:
                    if isinstance(static_x, list):
                        static_x_batch = static_x[start_idx:end_idx]
                    else:
                        static_x_batch = static_x[start_idx:end_idx]
                
                times_batch = None
                if times is not None:
                    if isinstance(times, list):
                        times_batch = times[start_idx:end_idx]
                    else:
                        times_batch = times[start_idx:end_idx]
                
                # Preprocess
                x_tensor, static_x_tensor, lens_tensor, times_tensor = \
                    self._preprocess_test_data(x_batch, static_x_batch, lens_batch, times_batch)
                
                # Forward pass
                pred, _, _, _, _ = self.forward(x_tensor, static_x_tensor, lens_tensor, times_tensor)
                # pred: (batch, num_time_bins + 1) - softmax probabilities
                
                # Calculate cumulative event probability F(t) = sum_{k=0}^{t} P(k)
                event_probs = pred[:, :-1]  # (batch, num_time_bins)
                cumsum_event_probs = torch.cumsum(event_probs, dim=1)  # (batch, num_time_bins)
                
                # Survival probability S(t) = 1 - F(t)
                survival_probs = 1.0 - cumsum_event_probs  # (batch, num_time_bins)
                
                # Extract survival probability at requested time points
                # For each requested time t_i, get survival_probs[:, t_bins[i]]
                batch_survival = survival_probs[:, t_bins].cpu().numpy()
                all_survival_probs.append(batch_survival)
        
        return np.concatenate(all_survival_probs, axis=0)  # (n_samples, len(t))

    def predict_risk(self, x, t, static_x=None, lens=None, times=None, risk=1, bs=100):
        """Predict risk (cumulative incidence) at given time points.
        
        Aligns with DDH's predict_risk behavior:
        risk(t) = 1 - S(t) = F(t) = cumulative probability of event by time t
        
        Args:
            x: List of arrays or numpy array of sequences (batch, T, features)
            t: Time points at which to evaluate risk (array or list)
            static_x: Static features (batch, static_features) or None
            lens: Sequence lengths (batch,) or None
            times: Time embeddings (batch, T, time_dim) or None
            risk: Risk index (default 1)
            bs: Batch size for prediction
            
        Returns:
            risk_probs: (batch, len(t)) array of risk probabilities (cumulative incidence) at specified times
        """
        # Risk = Cumulative event probability = 1 - Survival
        # This aligns with DDH: predict_risk returns 1 - predict_survival
        survival_probs = self.predict_survival(x, t, static_x, lens, times, risk, bs)
        return 1.0 - survival_probs





class CliTsSurv_head(nn.Module):
    """DeepHit head: outputs discrete probability distribution over time bins"""
    def __init__(self, feature_in, hidden_dims=[], num_time_bins=10):
        super(CliTsSurv_head, self).__init__()
        
        self.num_time_bins = num_time_bins
        
        hidden_dims = [feature_in] + hidden_dims
        layers = []
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        
        # Output: probability distribution over time bins + survival probability
        # Last bin is "survival" (no event observed)
        self.out_layer = nn.Linear(hidden_dims[-1], num_time_bins + 1)
        
    def forward(self, x):
        logits = self.out_layer(self.layers(x))
        # Softmax to get probability distribution
        return torch.softmax(logits, dim=-1)
    
