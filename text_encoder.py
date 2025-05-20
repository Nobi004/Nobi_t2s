import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self,vocab_size=100,d_model=256,nhead=4,num_layers=4,dim_forward=1024):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_forward=dim_forward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        
    def forward(self,x):
        x = self.embedding(x)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        return x.permute(1,0,2)
    
    