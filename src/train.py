import torch
import torch.nn as nn
from model import EndToEndTTS
from dataloader import get_dataloader, TTSDataset
import scipy.io.wavfile

def stft_loss(pred, target, sr=22050, n_fft=[1024, 2048, 512], hop_length=256):
    loss = 0
    for fft_size in n_fft:
        stft_pred = torch.stft(pred, n_fft=fft_size, hop_length=hop_length, return_complex=True)
        stft_target = torch.stft(target, n_fft=fft_size, hop_length=hop_length, return_complex=True)
        loss += torch.mean((stft_pred.abs() - stft_target.abs())**2)
    return loss / len(n_fft)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset to get vocab_size
    dataset = TTSDataset('data/preprocessed')
    vocab_size = len(dataset.phoneme_to_idx)
    print(f"Using vocab_size: {vocab_size}")
    
    model = EndToEndTTS(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_blocks=30,
        channels=64
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    mse_loss = nn.MSELoss()
    
    dataloader = get_dataloader('data/preprocessed', batch_size=8)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            try:
                phonemes = batch['phonemes'].to(device)
                waveform = batch['waveform'].to(device)
                durations = batch['durations'].to(device)
                pitch = batch['pitch'].to(device)
                energy = batch['energy'].to(device)
                
                print(f"Batch {batch_idx}: phonemes shape {phonemes.shape}, waveform shape {waveform.shape}")
                
                optimizer.zero_grad()
                waveform_pred, D_pred, P_pred, E_pred = model(phonemes, durations, pitch, energy)
                
                loss_waveform = stft_loss(waveform_pred, waveform)
                loss_duration = mse_loss(D_pred, durations)
                loss_pitch = mse_loss(P_pred, pitch)
                loss_energy = mse_loss(E_pred, energy)
                total_loss_batch = loss_waveform + loss_duration + loss_pitch + loss_energy
                
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        with open('training.log', 'a') as f:
            f.write(f"Epoch {epoch+1}, Loss: {avg_loss}\n")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train()