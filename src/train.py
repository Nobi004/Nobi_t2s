import torch 
import torch.nn as nn

# from src.model import End2EndTTS
# from src.dataloader import get_dataloader
from model import End2EndTTS
from dataloader import get_dataloader
import scipy.io.wavfile

def stft_loss(pred,target,sr=22050,n_fft=[1024,2048,512],hop_length=256):
    loss = 0
    for fft_size in n_fft:
        stft_pred = torch.stft(pred,n_fft=n_fft,hop_length=hop_length,return_complex=True)
        stft_target = torch.stft(target,n_fft=n_fft,hop_length=hop_length,return_complex=True)
        loss += torch.mean((stft_pred.abs() - stft_target.abs())**2)
        return loss / len(n_fft)
    
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = End2EndTTS(vocab_size=100,d_model=256,nhead=4,num_encoder_layers=4,num_blocks=30,channels=54)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    mse_loss = nn.MSELoss()

    dataloader = get_dataloader('data/preprocessed', batch_size=32)
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in dataloader:
            phonemes = batch['phonemes'].to(device)
            waveform = batch['waveform'].to(device)
            durations = batch['durations'].to(device)
            pitch = batch['pitch'].to(device)
            energy = batch['energy'].to(device)

            optimizer.zero_grad()
            waveform_pred,D_pred,P_pred,E_pred = model(phonemes,durations,pitch,energy)
            loss_waveform = stft_loss(waveform_pred,waveform)
            loss_durations = mse_loss(D_pred,durations)
            loss_pitch = mse_loss(P_pred,pitch)
            loss_energy = mse_loss(E_pred,energy)
            total_loss = loss_waveform + loss_durations + loss_pitch + loss_energy

            total_loss.backward()
            optimizer.step()

            total_loss += total_loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train()