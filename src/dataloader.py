import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TTSDataset(Dataset):
    def __init__(self, data_dir,max_phoneme_len=100,max_waveform_len=20480,hop_length=256):
        self.data_dir = data_dir
        self.max_phoneme_len = max_phoneme_len
        self.max_waveform_len = max_waveform_len
        self.hop_length = hop_length
        self.audio_ids = [f.split('_')[0] for f in os.listdir(data_dir) if f.endswith('_phonemes.npy')]
        self.phoneme_to_idx = {p: i for i, p in enumerate(set(p for id in self.audio_ids for p in np.load(os.path.join(data_dir, f'{id}_phonemes.npy'), allow_pickle=True)))}
    
    def __len__(self):
        return len(self.audio_ids)
    
    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]
        phonemes = np.load(os.path.join(self.data_dir, f'{audio_id}_phonemes.npy'), allow_pickle=True)
        waveform = np.load(os.path.join(self.data_dir, f'{audio_id}_waveform.npy'))
        durations = np.load(os.path.join(self.data_dir, f'{audio_id}_durations.npy'))
        pitch = np.load(os.path.join(self.data_dir, f'{audio_id}_pitch.npy'))
        energy = np.load(os.path.join(self.data_dir, f'{audio_id}_energy.npy'))
        
        # Convert phonemes to indices
        phoneme_indices = [self.phoneme_to_idx[p] for p in phonemes]
        phoneme_len = len(phoneme_indices)
        if phoneme_len > self.max_phoneme_len:
            phoneme_indices = phoneme_indices[:self.max_phoneme_len]
            durations = durations[:self.max_phoneme_len]
        else:
            phoneme_indices += [self.phoneme_to_idx['<pad>']] * (self.max_phoneme_len - phoneme_len)
            durations = np.pad(durations, (0, self.max_phoneme_len - phoneme_len), mode='constant') 
        
        # Truncate or pad to fixed length for batching
        max_len = 20480  # ~1 second at 22050 Hz
        waveform = waveform[:max_len] if len(waveform) > max_len else np.pad(waveform, (0, max_len - len(waveform)))
        durations = durations[:len(phonemes)]
        pitch = pitch[:max_len//256]
        energy = energy[:max_len//256]
        
        return {
            'phonemes': phoneme_indices,
            'waveform': torch.tensor(waveform, dtype=torch.float),
            'durations': torch.tensor(durations, dtype=torch.float),
            'pitch': torch.tensor(pitch, dtype=torch.float),
            'energy': torch.tensor(energy, dtype=torch.float)
        }

def get_dataloader(data_dir, batch_size=8):
    dataset = TTSDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)