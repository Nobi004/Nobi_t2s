import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)  # [seq_len, batch, d_model]
        x = self.transformer_encoder(x)
        return x.permute(1, 0, 2)  # [batch, seq_len, d_model]
    
    