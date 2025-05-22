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
    optimizer_g = torch.optim.Adam(
        [p for n, p in model.named_parameters() if 'discriminator' not in n], lr=0.0001
    )
    optimizer_d = torch.optim.Adam(
        [p for n, p in model.named_parameters() if 'discriminator' in n], lr=0.0001
    )
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    dataloader = get_dataloader('data/preprocessed', batch_size=4)
    num_epochs = 100
    total_batches = num_epochs * len(dataloader)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss_g = 0
        total_loss_d = 0
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
                
                # Generator (EndToEndTTS without discriminator)
                optimizer_g.zero_grad()
                waveform_pred, D_pred, P_pred, E_pred, disc_score = model(phonemes, durations, pitch, energy)
                
                print(f" | waveform_pred shape {waveform_pred.shape}, D_pred shape {D_pred.shape}, "
                      f"P_pred shape {P_pred.shape}, E_pred shape {E_pred.shape}, "
                      f"disc_score shape {disc_score.shape}", end="")
                
                loss_waveform = stft_loss(waveform_pred, waveform) * 0.1
                loss_duration = mse_loss(D_pred, durations) * 1.0
                loss_pitch = mse_loss(P_pred, pitch) * 1.0
                loss_energy = mse_loss(E_pred, energy) * 1.0
                loss_adv = bce_loss(disc_score, torch.ones_like(disc_score)) * 0.1  # Generator wants D to predict "real"
                total_loss_g_batch = loss_waveform + loss_duration + loss_pitch + loss_energy + loss_adv
                total_loss_g_batch = torch.clamp(total_loss_g_batch, 0, 1e6)
                
                total_loss_g_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in model.named_parameters() if 'discriminator' not in n], max_norm=1.0
                )
                optimizer_g.step()
                
                # Discriminator
                optimizer_d.zero_grad()
                real_score = model.discriminator(waveform)
                fake_score = model.discriminator(waveform_pred.detach())
                loss_d_real = bce_loss(real_score, torch.ones_like(real_score))
                loss_d_fake = bce_loss(fake_score, torch.zeros_like(fake_score))
                total_loss_d_batch = (loss_d_real + loss_d_fake) * 0.5
                total_loss_d_batch = torch.clamp(total_loss_d_batch, 0, 1e6)
                
                total_loss_d_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in model.named_parameters() if 'discriminator' in n], max_norm=1.0
                )
                optimizer_d.step()
                
                total_loss_g += total_loss_g_batch.item()
                total_loss_d += total_loss_d_batch.item()
                print(f" | Losses: waveform={loss_waveform.item():.2f}, duration={loss_duration.item():.2f}, "
                      f"pitch={loss_pitch.item():.2f}, energy={loss_energy.item():.2f}, "
                      f"adv={loss_adv.item():.2f}, gen={total_loss_g_batch.item():.2f}, "
                      f"disc={total_loss_d_batch.item():.2f}", end="")
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                print(f"Shapes: phonemes {phonemes.shape}, durations {durations.shape}, "
                      f"pitch {pitch.shape}, energy {energy.shape}, "
                      f"waveform_pred {waveform_pred.shape if 'waveform_pred' in locals() else 'N/A'}, "
                      f"D_pred {D_pred.shape if 'D_pred' in locals() else 'N/A'}, "
                      f"P_pred {P_pred.shape if 'P_pred' in locals() else 'N/A'}, "
                      f"E_pred {E_pred.shape if 'E_pred' in locals() else 'N/A'}, "
                      f"disc_score {disc_score.shape if 'disc_score' in locals() else 'N/A'}")
                print(f"Skipping batch {batch_idx}")
                continue
        
        avg_loss_g = total_loss_g / len(dataloader)
        avg_loss_d = total_loss_d / len(dataloader)
        print(f"\nEpoch {epoch+1}, Generator Loss: {avg_loss_g}, Discriminator Loss: {avg_loss_d}")
        with open('training.log', 'a') as f:
            f.write(f"Epoch {epoch+1}, Generator Loss: {avg_loss_g}, Discriminator Loss: {avg_loss_d}\n")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train()