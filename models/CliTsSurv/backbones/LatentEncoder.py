import torch
from torch import nn
import os
import numpy as np


class LatentEnc(nn.Module):
    def __init__(self, feature_in, hidden_dims):
        super(LatentEnc, self).__init__()
        
        if len(hidden_dims) == 0:
            hidden_dims = [0]
        
        hidden_dims = [feature_in] + hidden_dims
        
        layers = []
        for i in range(1, len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
            
        self.out_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1])

    
    def forward(self, x):
        x = self.layers(x)
        out = self.out_layer(x)
        
        return out
            
            
        

class LatentDec(nn.Module):
    def __init__(self, feature_in, hidden_dims, conti_feature_out, bi_feature_out):
        super(LatentDec, self).__init__()
        
        # Build hidden layers from feature_in through all hidden_dims
        last_dim = feature_in
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.ReLU())
            last_dim = dim
        self.layers = nn.Sequential(*layers)
        
        # Output layers project from the last hidden dim (or feature_in if hidden_dims=[])
        self.conti_out_layer = nn.Linear(last_dim, conti_feature_out)
        if bi_feature_out > 0:
            self.bi_out_layer = nn.Sequential(
                nn.Linear(last_dim, bi_feature_out),
                nn.Sigmoid()
            )
        else:
            self.bi_out_layer = None
            
    
    def forward(self, x):
        z = self.layers(x)
        conti_out = self.conti_out_layer(z)
        if self.bi_out_layer is not None:
            bi_out = self.bi_out_layer(z)
        else:
            bi_out = None
        
        return conti_out, bi_out
    
    


class Discriminator(nn.Module):
    def __init__(self, feature_in, hidden_dims=[32, 64, 32]):
        super(Discriminator, self).__init__()
        
        hidden_dims = [feature_in] + hidden_dims
        layers = []
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

    
    
class LatentEncoder(nn.Module):
    def __init__(self, conti_feature_in, bi_feature_in, enc_dims, dec_dims, disc_dims=[], beta=1, loss_func=nn.functional.mse_loss):
        super(LatentEncoder, self).__init__()
        
        if len(enc_dims) == 0:
            enc_dims = [0]
        if len(dec_dims) == 0:
            dec_dims = [0]
        
        self.enc = LatentEnc(conti_feature_in+bi_feature_in, enc_dims)
        self.dec = LatentDec(enc_dims[-1], dec_dims, conti_feature_in, bi_feature_in)
        self.neck_dim = enc_dims[-1]
        
        if len(disc_dims) > 0:
            self.discriminator = Discriminator(self.neck_dim, disc_dims)
        else:
            self.discriminator = None
        
        self.loss_func = loss_func
        self.loss_D = nn.BCELoss()
        
        self.beta = beta
        
        
    def loss_fn(self, pred, true, mask=None, predD=None, labelD=None):
        if mask is None:
            loss_rec = self.loss_func(pred, true)
        else:
            loss_rec = self.loss_func(torch.masked_select(pred, mask), torch.masked_select(true, mask))
        
        if not predD is None and not labelD is None:
            loss_gan = self.loss_D(predD, labelD)
        else:
            loss_gan = 0
        
        return loss_rec + self.beta*loss_gan, loss_rec, loss_gan
        
        
    def forward(self, x):
        latent = self.enc(x)
        
        if self.discriminator is None:
            outD = None
        else:
            outD = self.discriminator(latent)
        
        conti_rec, bi_rec = self.dec(latent)
        if bi_rec is not None:
            rec = torch.concat([conti_rec, bi_rec], dim=-1)
        else:
            rec = conti_rec
        
        return latent, rec, outD
    
    
    def train_model(self, train_dataloader, test_dataloader, epochs=100, lr=0.0001, save_dir='./checkpoints'):
        
        self.train()
        
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        optimizer = torch.optim.Adam([{'params': self.enc.parameters()}, {'params': self.dec.parameters()}], lr=lr)
        if self.discriminator is not None:
            optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        
        print('start training ...')
        best_val_loss = float('inf')
        best_epoch = 0
        loss_hist = []
        for epoch in range(epochs):
            
            self.train()
            epoch_losses = []
            
            for step, (x, y, mask) in enumerate(train_dataloader):
                
                if not self.discriminator is None:
                    latent, rec, predG = self.forward(x)
                    labelG = torch.zeros(predG.shape, device=x.device)
                    lossG = self.loss_D(predG, labelG)
                    
                    noise = torch.randn(latent.shape, device=x.device)
                    predN = self.discriminator(noise)
                    labelN = torch.ones(predN.shape, device=x.device)
                    lossN = self.loss_D(predN, labelN)
                    
                    optimizerD.zero_grad()
                    lossD = lossG + lossN
                    lossD.backward()
                    optimizerD.step()

                latent, rec, predD = self.forward(x)                
                optimizer.zero_grad()           # clear gradients for this training step
                if not predD is None:
                    labelD = torch.ones(predG.shape, device=x.device)
                    loss, loss_rec, loss_gan = self.loss_fn(rec, y, mask, predD, labelD)
                else:
                    loss, loss_rec, loss_gan = self.loss_fn(rec, y, mask)       
                loss.backward()                 # backpropagation, compute gradients
                loss_hist.append(loss.detach().cpu().numpy().astype(str))
                epoch_losses.append(loss.detach().cpu().numpy())
                optimizer.step()
                
                if (step+1) % 100 == 0:
                    if self.discriminator is not None:
                        print('[Epoch %d | %d] discriminator loss: %.4f (gen: %.4f, noise: %.4f) || train loss: %.4f (loss_rec: %.4f, loss_gan: %.4f)' 
                              % (epoch, step+1, lossD.detach().cpu().numpy(), lossG.detach().cpu().numpy(), lossN.detach().cpu().numpy(), loss.detach().cpu().numpy(), loss_rec.detach().cpu().numpy(), loss_gan.detach().cpu().numpy()))
                    else:
                        print('[Epoch %d | %d] train loss: %.4f (loss_rec: %.4f)'
                              % (epoch, step+1, loss.detach().cpu().numpy(), loss_rec.detach().cpu().numpy()))
            

            val_loss = self.test_model(test_dataloader)
            epoch_train_loss = np.mean(epoch_losses)
            print('[Epoch %d] train loss: %.4f || val loss: %.4f' % (epoch, epoch_train_loss, val_loss))
            
            if val_loss < best_val_loss:
                torch.save(self.state_dict(), os.path.join(save_dir, 'best.pt'))
                best_val_loss = val_loss
                best_epoch = epoch
                print('update best val loss %.4f, model saved' % best_val_loss)

            if (epoch+1) % 10 == 0:
                torch.save(self.state_dict(), os.path.join(save_dir, 'epoch_%d.pt'%epoch))
        
        print('training finished', '='*50)
        print('best val loss %.4f at epoch %d' % (best_val_loss, best_epoch))
        with open(os.path.join(save_dir, 'loss_hist'), 'w+', encoding='utf8', newline='') as f:
            loss_hist = list(map(lambda x: str(x)+'\n', loss_hist))
            f.writelines(loss_hist)
            
        self.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt')))
        
        
        
        
    def test_model(self, test_dataloader):
        
        self.eval()
        
        losses = []
        with torch.no_grad():
            for x, y, mask in test_dataloader:
                latent, rec, predD = self.forward(x)
                if predD is not None:
                    labelD = torch.ones(predD.shape, device=x.device)
                    loss, _, _ = self.loss_fn(rec, y, mask, predD, labelD)
                else:
                    loss, _, _ = self.loss_fn(rec, y, mask)
                losses.append(loss.detach().cpu().numpy())
        
        return np.mean(losses)