import numpy as np
import torch
import pandas as pd

def positional_encoding(positions, dim):
    angle_rads = get_angles(positions[:, np.newaxis], np.arange(dim)[np.newaxis: ], dim)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads
    
    return pos_encoding


def get_angles(pos, i, dim):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dim))
    return pos * angle_rates



def collate_fn(batch, infer=False, fillna=0, device='cuda'):
    """
    DeepHit version collate function for survival analysis
    
    Returns: (X, static_X, time_to_event, event_indicator, lengths, times, mask_nan, mask_nan_static)
    """
    static_X = torch.from_numpy(np.array([item[1] for item in batch]).astype(np.float32))
    time_to_event = torch.from_numpy(np.array([item[2] for item in batch]).astype(np.float32))
    event_indicator = torch.from_numpy(np.array([item[3] for item in batch]).astype(np.float32))
    lengths = [len(item[0]) for item in batch]
    times = [item[4] for item in batch]
    
    max_len = np.max(lengths)
    
    X = []
    times_padded = []
    for item in batch:
        x = np.array(item[0])
        time = np.array(item[4])
        if len(x) < max_len:   
            pad = np.zeros((max_len-x.shape[0], x.shape[1]))
            time_pad = np.zeros((max_len-time.shape[0], time.shape[1]))
            x = np.vstack([x, pad])
            time = np.vstack([time, time_pad])
        X.append(x)
        times_padded.append(time)
    
    times_tensor = torch.from_numpy(np.array(times_padded).astype(np.float32))
    X = torch.from_numpy(np.array(X).astype(np.float32))    
    mask_nan = torch.where(torch.isnan(X), True, False)
    mask_nan_static = torch.where(torch.isnan(static_X), True, False)
    
    X = torch.masked_fill(X, mask_nan, fillna)
    static_X = torch.masked_fill(static_X, mask_nan_static, fillna)
    
    X = X.to(device)
    static_X = static_X.to(device)
    time_to_event = time_to_event.to(device)
    event_indicator = event_indicator.to(device)
    times_tensor = times_tensor.to(device)
    mask_nan = torch.logical_not(mask_nan).to(device)
    mask_nan_static = torch.logical_not(mask_nan_static).to(device)
    
    if infer:
        ids = [item[5] for item in batch]
        return X, static_X, time_to_event, event_indicator, lengths, times_tensor, mask_nan, mask_nan_static, ids
        
    return X, static_X, time_to_event, event_indicator, lengths, times_tensor, mask_nan, mask_nan_static


def collate_fn_latentenc(batch, device='cuda'):
    X = torch.concat([torch.unsqueeze(item[0], dim=0) for item in batch]).to(device)
    Y = torch.concat([torch.unsqueeze(item[1], dim=0) for item in batch]).to(device)
    mask = torch.concat([torch.unsqueeze(item[2], dim=0) for item in batch]).to(device)
    
    return X, Y, mask


def norm_df_ignorena(df: pd.DataFrame, cols, means=None, stds=None):
    
    if means is None or stds is None:
        means = np.zeros(len(cols))
        stds = np.zeros(len(cols))
        for i, col in enumerate(cols):
            mean = df.loc[pd.notnull(df.loc[:, col]), col].mean()
            std = df.loc[pd.notnull(df.loc[:, col]), col].std()
            df.loc[pd.notnull(df.loc[:, col]), col] = (df.loc[pd.notnull(df.loc[:, col]), col] - mean) / std
            means[i] = mean
            stds[i] = std
    else:
        assert len(cols) == len(means), '[Error] given num means doesnot match num features'
        assert len(cols) == len(stds), '[Error] given num stds doesnot match num features'

        for i, col in enumerate(cols):
            df.loc[pd.notnull(df.loc[:, col]), col] = (df.loc[pd.notnull(df.loc[:, col]), col] - means[i]) / stds[i]

    return df, means, stds

def normalize_df(df: pd.DataFrame, cols, means=None, stds=None):
    
    if means is None or stds is None: 
        means = df.loc[:, cols].mean()
        stds = df.loc[:, cols].std()
    
    df.loc[:, cols] = (df.loc[:, cols] - means) / stds
    
    return df, means, stds



