import torch
import torch.nn as nn
from model import EndToEndTTS
from dataloader import get_dataloader, TTSDataset
import scipy.io.wavfile
import sys

def stft_loss(pred, target, sr=22050, n_fft=[1024, 2048, 512], hop_length=256):
    loss = 0
    for fft_size in n_fft:
        stft_pred = torch.stft(pred, n_fft=fft_size, hop_length=hop_length, return_complex=True)
        stft_target = torch.stft(target, n_fft=fft_size, hop_length=hop_length, return_complex=True)
        loss += torch.mean((stft_pred.abs() - stft_target.abs())**2)
    return loss / len(n_fft)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    num_epochs = 100
    total_batches = num_epochs * len(dataloader)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            current_batch = epoch * len(dataloader) + batch_idx + 1
            progress_percent = (current_batch / total_batches) * 100
            
            try:
                phonemes = batch['phonemes'].to(device)
                waveform = batch['waveform'].to(device)
                durations = batch['durations'].to(device)
                pitch = batch['pitch'].to(device)
                energy = batch['energy'].to(device)
                
                print(f"\rTraining Progress: {progress_percent:.2f}% | "
                      f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"phonemes shape {phonemes.shape}, waveform shape {waveform.shape}, "
                      f"durations shape {durations.shape}, pitch shape {pitch.shape}, "
                      f"energy shape {energy.shape}", end="")
                
                optimizer.zero_grad()
                waveform_pred, D_pred, P_pred, E_pred = model(phonemes, durations, pitch, energy)
                
                print(f" | waveform_pred shape {waveform_pred.shape}, D_pred shape {D_pred.shape}, "
                      f"P_pred shape {P_pred.shape}, E_pred shape {E_pred.shape}", end="")
                
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
                print(f"\nError in batch {batch_idx}: {e}")
                print(f"Shapes: phonemes {phonemes.shape}, durations {durations.shape}, "
                      f"pitch {pitch.shape}, energy {energy.shape}, "
                      f"waveform_pred {waveform_pred.shape if 'waveform_pred' in locals() else 'N/A'}, "
                      f"D_pred {D_pred.shape if 'D_pred' in locals() else 'N/A'}, "
                      f"P_pred {P_pred.shape if 'P_pred' in locals() else 'N/A'}, "
                      f"E_pred {E_pred.shape if 'E_pred' in locals() else 'N/A'}")
                print(f"Skipping batch {batch_idx}")
                continue
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}, Loss: {avg_loss}")
        with open('training.log', 'a') as f:
            f.write(f"Epoch {epoch+1}, Loss: {avg_loss}\n")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train()