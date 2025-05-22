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
        
        # Handle waveform
        waveform = waveform[:self.max_waveform_len] if len(waveform) > self.max_waveform_len else np.pad(waveform, (0, self.max_waveform_len - len(waveform)))

        # Handle pitch and energy
        max_frames = self.max_waveform_len // self.hop_length
        pitch = pitch[:max_frames] if len(pitch) > max_frames else np.pad(pitch,(0,max_frames - len(pitch)))
        energy = energy[:max_frames] if len(energy) > max_frames else np.pad(energy,(0,max_frames - len(energy)))

        
        return {
            'phonemes': torch.tensor(phoneme_indices,dtype=torch.long),
            'waveform': torch.tensor(waveform, dtype=torch.float),
            'durations': torch.tensor(durations, dtype=torch.float),
            'pitch': torch.tensor(pitch, dtype=torch.float),
            'energy': torch.tensor(energy, dtype=torch.float),
            'phoneme_len': phoneme_len
        }
def collate_fn(batch):
    phonemes = torch.stack([item['phonemes'] for item in batch])
    waveforms = torch.stack([item['waveform'] for item in batch])
    durations = torch.stack([item['durations'] for item in batch])
    pitch = torch.stack([item['pitch'] for item in batch])
    energy = torch.stack([item['energy'] for item in batch])
    phoneme_lens = torch.tensor([item['phoneme_len'] for item in batch] , dtype=torch.long)
    return {
        'phonemes': phonemes,
        'waveform': waveforms,
        'durations': durations,
        'pitch': pitch,
        'energy': energy,
        'phoneme_lens': phoneme_lens
    }

def get_dataloader(data_dir, batch_size=8):
    dataset = TTSDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn)