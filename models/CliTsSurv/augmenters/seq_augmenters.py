import torch
from torch import nn
import numpy as np


def build_mask(lens, T, device='cuda'):
    masks = []
    for len in lens:
        mask = np.zeros(T)
        mask[:len] += 1
        masks.append(torch.from_numpy(mask.astype(int)))
    mask = torch.concatenate(masks)
    mask = torch.where(mask==1, True, False)
    mask = mask.to(device)
    return mask



class AddGaussianNoise(nn.Module):
    def __init__(self, p=0.3, device='cuda'):
        super(AddGaussianNoise, self).__init__()
        
        self.p = p
        self.device=device
        
    def forward(self, z, lens, times, labels):
        
        B, T, K = z.shape
        
        # build noise
        p = torch.rand(B)
        noise = torch.zeros_like(z)
        for i in range(B):
            if p[i] < self.p:
                idx = np.random.choice(lens[i], size=1)
                mean = torch.zeros(K).to(self.device)
                std = torch.ones(K).to(self.device) / 10
                noise[i, idx] = torch.normal(mean, std)

        z = z + noise
       
        return z, lens, times
            
            
            
class InnerInterpolate(nn.Module):
    def __init__(self, p=0.3, device='cuda'):
        super(InnerInterpolate, self).__init__()
        
        self.p = p
        self.device=device
        
        
    def forward(self, z, lens, times, labels):
        
        B, T, K = z.shape
    
        # build noise
        p = torch.rand(B)
        z_ = torch.zeros_like(z).to(self.device)
        times_ = torch.zeros_like(times).to(self.device)
        for i in range(B):
            # do nothing
            if p[i] > self.p or lens[i] >= T or lens[i] <= 1:
                z_[i] = z[i]
                times_[i] = times[i]
            # compute interpolation
            else:
                idx = np.random.choice(lens[i]-1, size=1)[0]
                ratio = torch.rand(1).to(self.device)
                
                z_[i, :idx+1] = z[i, :idx+1]
                z_[i, idx+1] = ratio*z[i, idx] + (1-ratio)*z[i, idx+1]
                z_[i, idx+2:] = z[i, idx+1:-1]
                
                times_[i, :idx+1] = times[i, :idx+1]
                times_[i, idx+1] = ratio*times[i, idx] + (1-ratio)*times[i, idx+1]
                times_[i, idx+2:] = times[i, idx+1:-1]
                
                lens[i] += 1
                
        z = z_
        times = times_
       
        return z, lens, times
            
            
            
            
class RandomRemove(nn.Module):
    def __init__(self, p=0.3, device='cuda'):
        super(RandomRemove, self).__init__()
        
        self.p = p
        self.device=device
        
    def forward(self, z, lens, times, labels):
        
        B, T, K = z.shape
        
        # build noise
        p = torch.rand(B)
        z_ = torch.zeros_like(z).to(self.device)
        times_ = torch.zeros_like(times).to(self.device)
        for i in range(B):
            # do nothing
            if p[i] > self.p or lens[i] <= 1:
                z_[i] = z[i]
                times_[i] = times[i]
            # compute interpolation
            else:
                idx = np.random.choice(lens[i], size=1)[0]
                
                z_[i, :idx] = z[i, :idx]
                z_[i, idx:lens[i]-1] = z[i, idx+1:lens[i]]
                
                times_[i, :idx] = times[i, :idx]
                times_[i, idx:lens[i]-1] = times[i, idx+1:lens[i]]
                
                lens[i] -= 1
                
        z = z_
        times = times_
       
        return z, lens, times
    
    
class RandomSlice(nn.Module):
    def __init__(self, p=0.3, device='cuda'):
        super(RandomSlice, self).__init__()
        
        self.p = p
        self.device=device
        
    def forward(self, z, lens, times, labels):
        
        B, T, K = z.shape
        
        # build noise
        p = torch.rand(B)
        z_ = torch.zeros_like(z).to(self.device)
        times_ = torch.zeros_like(times).to(self.device)
        for i in range(B):
            # do nothing
            if p[i] > self.p or lens[i] <= 1:
                z_[i] = z[i]
                times_[i] = times[i]
            # compute interpolation
            else:
                idx = np.random.choice(lens[i], size=1)[0]
                idx_end = np.random.choice(range(idx+1, lens[i]+1), size=1)[0]
                
                lens[i] = idx_end - idx
                z_[i, :lens[i]] = z[i, idx:idx_end]
                times_[i, :lens[i]] = times[i, idx:idx_end] - times[i, idx]
                
        z = z_
        times = times_
       
        return z, lens, times