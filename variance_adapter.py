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
        x = self.relu(self.comv1(x))
        x = self.relu(self.conv2(x))
        x = self.linear(x.permute(0,2,1)).squeeze(-1)
        return x 
    
