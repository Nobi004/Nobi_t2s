import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,vocab_size, d_model,nhead=4,num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,dim_feadforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers)

        def forward(self,x):
            x = self.embedding(x) 
            x = x.permute(1,0,2)
            x = self.transformer_encoder(x)
            return x.permute(1,0,2)
        
class DurationPredictor(nn.Module):
    def __init__(self,in_dim,filter_size,kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim,filter_size,kernel_size,padding=1)
        self.conv2 = nn.Conv1d(filter_size,filter_size,kernel_size,padding=1)
        self.linear = nn.Linear(filter_size,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.linear(x.permute(0,2,1).squeeze(1))
        return x 
    
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
        x = self.linear(x.permute(0,2,1).squeeze(1))
        return x
    
class EnergyPredictor(nn.Module):
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
        x = self.linear(x.permute(0,2,1).squeeze(1))
        return x


class


  
