import numpy as np
audio_id = 'LJ001-0001'
phonemes = np.load(f'data/preprocessed/{audio_id}_phonemes.npy', allow_pickle=True)
durations = np.load(f'data/preprocessed/{audio_id}_durations.npy')
pitch = np.load(f'data/preprocessed/{audio_id}_pitch.npy')
energy = np.load(f'data/preprocessed/{audio_id}_energy.npy')
print(f"Phonemes len: {len(phonemes)}, Durations shape: {durations.shape}, "
      f"Pitch shape: {pitch.shape}, Energy shape: {energy.shape}")