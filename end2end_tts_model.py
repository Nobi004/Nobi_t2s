import torch
import torch.nn as nn

from waveform_decoder import WaveformDecoder
from text_encoder import TExtEncoder        # assuming "Encoder" is in encoder_module.py
from variance_adaptor import VarianceAdaptor  # assuming "VarianceAdaptor" is in variance_adaptor.py
class End2EndTTSModel(nn.Module):
    def __init__(self,vocab_size=100,d_model=256,nhead=4,num_encoder_layers=4,num_blocks=30,channels=64):
        super().__init__()
        self.encoder = Encoder(vocab_size,d_model,nhead,num_encoder_layers)
        self.variance_adaptor = VarianceAdaptor(d_model)
        self.waveform_decoder = WaveformDecoder(d_model,num_blocks,channels)
        
    def forward(self,text,D_gt=None,P_gt=None,E_gt=None,is_inference=False):
        H = self.encoder(text)
        H_adapted , D_pred,P_pred,E_pred = self.variance_adaptor(H,D_gt,P_gt,E_gt,is_inference)
        waveform = self.waveform_decoder(H_adapted)
        return waveform,D_pred,P_pred,E_pred

    