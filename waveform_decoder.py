import torch 
import torch.nn as nn

class GateDilatedConv(nn.Module):
    def __init__(self,channels,kernel_size,dilation):
        super().__init__()
        padding = (kernel_size -1) * dilation // 2 
        self.conv = nn.Conv1d(channels,2*channels,kernel_size,dilation=dilation,padding=padding)
        
        
    def forward(self,x):
        z=self.conv(x)
        z1,z2 = z.chunk(2.dim=1)
        return torch.tanh(z1)*torch.sigmoid(z2)
    
class ResidualBlock(nn.Module):
    def __init__(self,channels,kernel_size,dilation):
        super().__init__()
        self.gated_conv = GateDilatedConv(channels,kernel_size,dilation)
        self.conv1x1 = nn.Conv1d(channels,channels,kernel_size=1)
        
    def forward(self,x):
        out=self.gated_conv(x)
        residual = self.conv1x1(x)
        return x + residual


        