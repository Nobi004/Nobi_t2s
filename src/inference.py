import os
preprocessed_dir = 'data/preprocessed'
phoneme_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('_phonemes.npy')]
print(f"Found {len(phoneme_files)} phoneme files")