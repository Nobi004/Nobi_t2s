import torch 
import torch.nn as nn

from src.model import End2EndTTS
from src.dataloader import get_dataloader
import scipy.io.wavfile

def stft_loss(pred,target,sr=22050,n_fft=[1024,2048,512],hop_length=256):
    loss = 0
    for fft_size in n_fft:
        stft_pred = torch.stft(pred,n_fft=n_fft,hop_length=hop_length,return_complex=True)
        stft_target = torch.stft(target,n_fft=n_fft,hop_length=hop_length,return_complex=True)
        loss += torch.mean((stft_pred.abs() - stft_target.abs())**2)
        return loss / len(n_fft)
    
