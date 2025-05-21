"""Adversarial Training (Optional)
Implement a discriminator for adversarial training to improve waveform quality.
Use a structure similar to Parallel WaveGAN with 10 layers of dilated convolutions
"""
import torch
import torch.nn as nn

# Define the Discriminator class
# This class is a simple implementation of a discriminator network
class Discriminator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, channels, kernel_size=3, padding=1)
            for _ in range(10)
        ])
        self.final = nn.Conv1d(channels, 1, kernel_size=1)
  
    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.final(x)


