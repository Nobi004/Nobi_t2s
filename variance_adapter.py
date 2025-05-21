"""Variance Adaptor
Adds duration, pitch, and energy to the hidden representation to address the one-to-many 
mapping problem (multiple speech variations for one text).
Includes predictors for each variance and projects them to the hidden dimension"""

import torch
import torch.nn as nn

#Duration Predictor
class DurationPredictor(nn.Module):
    def __init__(self,in_dim=256,filter_size=256,kernel_size=3):
        
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim,filter_size,kernel_size=kernel_size,padding=1)
        self.conv2 = nn.Conv1d(filter_size,filter_size,kernel_size,padding=1)
        self.linear = nn.Linear(filter_size,1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x =x.permute(0,2,1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.linear(x.permute(0,2,1)).squeeze(-1)
        return x 
    
#Pitch Predictor
class PitchPredictor(nn.Module):
    def __init__(self,in_dim=256,filter_size=256,kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim,filter_size,kernel_size,padding=1)
        self.conv2 = nn.Conv1d(filter_size,filter_size,kernel_size,padding=1)
        self.linear = nn.Linear(filter_size,1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        self.linear(x.permute(0,2,1)).squeeze(-1)
        return x
    
# Variation Adaptor
class VarianceAdaptor(nn.Module):
    def __init__(self,d_model=256):
        super().__init__()
        self.duration_predictor = DurationPredictor(d_model)
        self.pitch_predictor = PitchPredictor(d_model)
        self.energy_predictor = PitchPredictor(d_model) # same structure as pitch predictor
        self.pitch_proj = nn.Linear(1,d_model)
        self.energy_proj = nn.Linear(1,d_model)
        
        
    def expand_hidden(self,H,D):
        batch_size,seq_len,hidden_dim , d_model = H.size()
        D = D.long()
        max_T = D.sum(dim=1).max()
        H_expanded = torch.zeros(batch_size,max_T,d_model).to(H.device)
        for b in range(batch_size):
            indices = torch.repeat_interleave(torch.arange(seq_len),device=H.device)
            H_expanded[b,:len(indices)] = H[b,indices]
            
        return H_expanded
    
    def forward(self,H,D_gt=None,p_gt=None,E_gt=None,is_inference=False):
        D_pred = self.duration_predictor(H)
        D = D_pred if is_inference else D_gt
        H_expanded = self.expand_didden(H,D)
        P_pred = self.pitch_predictor(H_expanded)
        E_pred = self.energy_predictor(H_expanded)
        P = P_pred if is_inference else P_gt
        E = E_pred if is_inference else E_gt
        P_proj = self.pitch_proj(P.unsqueeze(-1))
        E_proj = self.energy_proj(E.unsqueeze(-1))
        H_adapted = H_expanded + P_proj+E_proj
        return H_adapted,D_pred,P_pred,E_pred

        
        
        
        