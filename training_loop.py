import torch 
import torch.nn as nn
from end2end_tts_model import End2End_TTSModel
from data_loader import DataLoader
from loss_fn import sifi_loss, mse_loss

# Initialize the data loader
dataloader = DataLoader(batch_size=32, num_workers=4)
model = End2End_TTSModel()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    for i, waveform,D_gt,P_gt,E_gt in dataloader:
        optimizer.zero_grad()
        waveform_pred, D_pred, P_pred, E_pred = model(text,D_gt,P_gt,E_gt)
        loss_waveform = mse_loss(waveform_pred,waveform)
        loss_duration = mse_loss(D_pred,D_gt)
        loss_pitch = mse_loss(P_pred,P_gt)
        loss_energy = mse_loss(E_pred,E_gt)
        total_loss = loss_waveform + loss_duration + loss_pitch + loss_energy
        total_loss.backward()
        optimizer.step()