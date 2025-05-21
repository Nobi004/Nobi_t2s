import os
import numpy as np
import scipy.io.wavfile
from scipy.signal import correlate

def load_cmudict(file_path):
    cmudict = {}
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;'): continue
            parts = line.strip().split()
            word = parts[0].lower()
            phonemes = parts[1:]
            cmudict[word] = phonemes
    return cmudict

def simple_g2p(text, cmudict):
    words = text.lower().split()
    phonemes = []
    for word in words:
        if word in cmudict:
            phonemes.extend(cmudict[word])
        else:
            phonemes.extend([c.upper() for c in word if c.isalpha()])
    return phonemes

def extract_pitch(audio, sr=22050, frame_length=1024, hop_length=256):
    frames = [audio[i:i+frame_length] for i in range(0, len(audio)-frame_length, hop_length)]
    f0 = []
    for frame in frames:
        corr = correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        peak = np.argmax(corr[1:]) + 1
        f0.append(sr / peak if peak > 0 and peak < len(corr) else 0)
    return np.array(f0)

def extract_energy(audio, frame_length=1024, hop_length=256):
    frames = [audio[i:i+frame_length] for i in range(0, len(audio)-frame_length, hop_length)]
    energy = [np.sqrt(np.mean(frame**2)) for frame in frames]
    return np.array(energy)

def simple_dtw(audio, phonemes):
    duration_per_phoneme = len(audio) // (len(phonemes) * (22050 // 256))
    durations = np.array([duration_per_phoneme] * len(phonemes))
    return durations

def preprocess_dataset(data_dir, output_dir, cmudict_path):
    cmudict = load_cmudict(cmudict_path)
    os.makedirs(output_dir, exist_ok=True)
    
    metadata_file = os.path.join(data_dir, 'metadata.csv')
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split('|')
        audio_id, _, text = parts
        audio_path = os.path.join(data_dir, 'wavs', f'{audio_id}.wav')
        
        # Load and normalize audio
        sr, audio = scipy.io.wavfile.read(audio_path)
        if sr != 22050:
            raise ValueError(f"Expected sample rate 22050, got {sr}")
        audio = audio / np.max(np.abs(audio))
        
        # Extract features
        phonemes = simple_g2p(text, cmudict)
        durations = simple_dtw(audio, phonemes)
        pitch = extract_pitch(audio)
        energy = extract_energy(audio)
        
        # Save preprocessed data
        np.save(os.path.join(output_dir, f'{audio_id}_phonemes.npy'), phonemes)
        np.save(os.path.join(output_dir, f'{audio_id}_waveform.npy'), audio)
        np.save(os.path.join(output_dir, f'{audio_id}_durations.npy'), durations)
        np.save(os.path.join(output_dir, f'{audio_id}_pitch.npy'), pitch)
        np.save(os.path.join(output_dir, f'{audio_id}_energy.npy'), energy)

if __name__ == "__main__":
    preprocess_dataset('data/LJSpeech-1.1', 'data/preprocessed', 'cmudict-0.7b')