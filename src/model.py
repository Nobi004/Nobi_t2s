import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,vocab_size, d_model,nhead=4,num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

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


class VarianceAdaptor(nn.Module):
    def __init__(self, in_dim=256, filter_size=256, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, filter_size, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=1)
        self.linear = nn.Linear(filter_size, 1)
        self.relu = nn.ReLU()

    def expand_hidden(self,H,D):
        batch_size,seq_len,d_model = H.size()
        D= D.long()
        max_T = D.sum(dim=1).max()
        H_expanded = torch.zeros(batch_size,max_T,d_model,device=H.device)
        for b in range(batch_size):
            indices = torch.repeat_interleave(torch.arrange(seq_len,device=H.device),D[b])
            H_expanded[b,:len(indices)] = H[b,indices]
            return H_expanded

    def forward(self,H,D_gt=None,P_gt=None,E_gt=None,is_inference=False):
        D_pred = self.duration_predictor(H)
        D = D_pred if is_inference else D_gt 
        H_expanded = self.expand_hidden(H,D)
        P_pred = self.pitch_predictor(H_expanded)
        E_pred = self.energy_predictor(H_expanded)
        P = P_pred if is_inference else P_gt
        E = E_pred if is_inference else E_gt
        P_proj = self.pitch_proj(P.unsqueeze(-1))
        E_proj = self.energy_proj(E.unsqueeze(-1))
        H_adapted = H_expanded + P_proj + E_proj
        return H_adapted, D_pred, P_pred, E_pred

class GateDialatedConv(nn.Module):
    def __init__(self,channels,kernel_size,dilation):
        super().__init__()
        padding = (kernel_size -1)* dilation // 2
        self.conv = nn.Conv1d(channels,2 *channels,kernel_size,dilation,padding)

    def forward(self,x):
        z = self.conv(x)
        z1,z2 = z.chunk(2,dim=1)
        return torch.tanh(z1) * torch.sigmoid(z2)
    
class ResidualBlock(nn.Module):
    def __init__(self,channels,kernel_size,dilation):
        super().__init__()
        self.gated_conv = GateDialatedConv(channels,kernel_size,dilation)
        self.conv1x1 = nn.Conv1d(channels,channels,kernel_size=1)

    def forward(self,x):
        out = self.gated_conv(x)
        residual = self.conv1x1(out)
        return x + residual
    

class WaveformDecoder(nn.Module):
    def __init__(self,in_channels=256,out_channels=1,num_blocks=30,channels=64):
        super().__init__()
        self.unsample = nn.ConvTranspose1d(in_channels,channels,kernel_size=256,stride=256)
        self.blocks = nn.ModuleList([
            ResidualBlock(channels,kernel_size=3,dilation=2**(i % 10))
            for i in range(num_blocks)
        ])
        self.final_conv = nn.Conv1d(channels,out_channels,kernel_size=1)

    def forward(self,x):
        x = self.unsample(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x.squeeze(1)
    
class End2EndTTS(nn.Module):
    def __init__(self,vocab_size=256,d_model=256,nhead=4,num_encoder_layers=4,num_blocks=30,channels=64):
        super().__init__()
        self.encoder = Encoder(vocab_size,d_model,nhead,num_encoder_layers)
        self.variance_adaptor = VarianceAdaptor(d_model)
        self.waveform_decoder = WaveformDecoder(d_model,num_blocks=num_blocks,channels=channels)

    def forward(self,text,D_gt=None,P_gt=None,E_gt=None,is_inference=False):
        H = self.encoder(text)
        H_adapted,D_pred,P_pred,E_pred = self.variance_adaptor(H,D_gt,P_gt,E_gt,is_inference)
        waveform = self.waveform_decoder(H_adapted)
        return waveform,D_pred,P_pred,E_pred
    

        


  
