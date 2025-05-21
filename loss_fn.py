import torch 
import numpy as np 

def sifi_loss(pred,target,sr=22050,n_fft=[1024,2048,512],hop_length=256):
    """
    Compute the SIFI loss between the predicted and target spectrograms.
    
    Args:
        pred (torch.Tensor): Predicted spectrogram of shape (batch_size, n_fft//2+1, time_steps).
        target (torch.Tensor): Target spectrogram of shape (batch_size, n_fft//2+1, time_steps).
        sr (int): Sampling rate. Default is 22050.
        n_fft (list): List of FFT sizes for different frequency bands. Default is [1024, 2048, 512].
        hop_length (int): Hop length for STFT. Default is 256.
    
    Returns:
        torch.Tensor: Computed SIFI loss.
    """
    # Compute the SIFI loss
    loss = 0
    for fft_size in n_fft:
        stft_pred = torch.stft(pred,n_fft=fft_size,hop_length=hop_length,return_complex=True)
        stft_target = torch.stft(target,n_fft=fft_size,hop_length=hop_length,return_complex=True)
        loss += torch.mean((stft_pred.abs() - stft_target.abs())**2)

    return loss / len(n_fft)


