import torch 
import torch.nn as nn

## Create a gated dilated convolution layer
# This layer is used in the residual block
class GateDilatedConv(nn.Module):
    def __init__(self,channels,kernel_size,dilation):
        super().__init__()
        padding = (kernel_size -1) * dilation // 2 
        self.conv = nn.Conv1d(channels,2*channels,kernel_size,dilation=dilation,padding=padding)
        
        
    def forward(self,x):
        z=self.conv(x)
        z1,z2 = z.chunk(2,dim=1)
        return torch.tanh(z1)*torch.sigmoid(z2)
    
# Create a residual block with gated dilated convolution
# and a 1x1 convolution
class ResidualBlock(nn.Module):
    def __init__(self,channels,kernel_size,dilation):
        super().__init__()
        self.gated_conv = GateDilatedConv(channels,kernel_size,dilation)
        self.conv1x1 = nn.Conv1d(channels,channels,kernel_size=1)
        
    def forward(self,x):
        out=self.gated_conv(x)
        residual = self.conv1x1(x)
        return x + residual

# Create a WaveformDecoder class

class WaveformDecoder(nn.Module):
    def __init__(self,in_channels=256,out_channels=1,num_blocks=30,channels=64):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels,channels,kernel_size=256,stride=256)
        self.blocks = nn.ModuleList([
            ResidualBlock(channels,kernel_size=3,dilation=2**(i % 10))
            for i in range(num_blocks)
        ])
        self.final_conv = nn.Conv1d(channels,out_channels,kernel_size=1)
        
    def forward(self,x):
        x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x.squeeze(1) # [batch, L]
        

        