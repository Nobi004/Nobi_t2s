import os
import numpy as np
import scipy.io.wavfile
from scipy.signal import correlate
from scipy.interpolate import interp1d

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

def extract_pitch(audio, sr=22050, frame_length=1024, hop_length=256, max_frames=80):
    frames = [audio[i:i+frame_length] for i in range(0, len(audio)-frame_length, hop_length)]
    f0 = []
    for frame in frames:
        corr = correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        peak = np.argmax(corr[1:]) + 1
        f0.append(sr / peak if peak > 0 and peak < len(corr) else 0)
    f0 = np.array(f0)[:max_frames]
    # Interpolate non-zero pitch values
    indices = np.arange(len(f0))
    valid = f0 > 0
    if np.sum(valid) > 1:
        f0_valid = f0[valid]
        indices_valid = indices[valid]
        interp_func = interp1d(indices_valid, f0_valid, bounds_error=False, fill_value=(f0_valid[0], f0_valid[-1]))
        f0 = interp_func(indices)
    f0 = np.clip(f0, 50, 500)  # Hz range
    f0 = np.log(f0 + 1e-8)  # Log-transform
    f0 = (f0 - np.mean(f0)) / (np.std(f0) + 1e-8)
    f0 = np.pad(f0, (0, max_frames - len(f0)), mode='constant') if len(f0) < max_frames else f0[:max_frames]
    return f0

def extract_energy(audio, frame_length=1024, hop_length=256, max_frames=80):
    frames = [audio[i:i+frame_length] for i in range(0, len(audio)-frame_length, hop_length)]
    energy = [np.sqrt(np.mean(frame**2)) for frame in frames]
    energy = np.array(energy)[:max_frames]
    energy = np.clip(energy, 1e-6, 1.0)
    energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-8)
    energy = np.pad(energy, (0, max_frames - len(energy)), mode='constant') if len(energy) < max_frames else energy
    return energy

def simple_dtw(audio, phonemes, max_phoneme_len=100):
    duration_per_phoneme = len(audio) // (len(phonemes) * (22050 // 256))
    durations = np.array([duration_per_phoneme] * len(phonemes))
    durations = durations[:max_phoneme_len]
    durations = np.clip(durations, 1, 100)
    durations = durations / (np.mean(durations) + 1e-8)
    durations = np.pad(durations, (0, max_phoneme_len - len(durations)), mode='constant') if len(durations) < max_phoneme_len else durations
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
        
        sr, audio = scipy.io.wavfile.read(audio_path)
        if sr != 22050:
            raise ValueError(f"Expected sample rate 22050, got {sr}")
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        audio = np.clip(audio, -1.0, 1.0)
        
        phonemes = simple_g2p(text, cmudict)
        durations = simple_dtw(audio, phonemes)
        pitch = extract_pitch(audio)
        energy = extract_energy(audio)
        
        np.save(os.path.join(output_dir, f'{audio_id}_phonemes.npy'), phonemes)
        np.save(os.path.join(output_dir, f'{audio_id}_waveform.npy'), audio)
        np.save(os.path.join(output_dir, f'{audio_id}_durations.npy'), durations)
        np.save(os.path.join(output_dir, f'{audio_id}_pitch.npy'), pitch)
        np.save(os.path.join(output_dir, f'{audio_id}_energy.npy'), energy)

if __name__ == "__main__":
    preprocess_dataset('data/LJSpeech-1.1', 'data/preprocessed', 'cmudict-0.7b')



    
# import os
# import numpy as np
# import scipy.io.wavfile
# from scipy.signal import correlate

# def load_cmudict(file_path):
#     cmudict = {}
#     with open(file_path, 'r', encoding='latin-1') as f:
#         for line in f:
#             if line.startswith(';;;'): continue
#             parts = line.strip().split()
#             word = parts[0].lower()
#             phonemes = parts[1:]
#             cmudict[word] = phonemes
#     return cmudict

# def simple_g2p(text, cmudict):
#     words = text.lower().split()
#     phonemes = []
#     for word in words:
#         if word in cmudict:
#             phonemes.extend(cmudict[word])
#         else:
#             phonemes.extend([c.upper() for c in word if c.isalpha()])
#     return phonemes

# def extract_pitch(audio, sr=22050, frame_length=1024, hop_length=256):
#     frames = [audio[i:i+frame_length] for i in range(0, len(audio)-frame_length, hop_length)]
#     f0 = []
#     for frame in frames:
#         corr = correlate(frame, frame, mode='full')
#         corr = corr[len(corr)//2:]
#         peak = np.argmax(corr[1:]) + 1
#         f0.append(sr / peak if peak > 0 and peak < len(corr) else 0)
#     return np.array(f0)

# def extract_energy(audio, frame_length=1024, hop_length=256):
#     frames = [audio[i:i+frame_length] for i in range(0, len(audio)-frame_length, hop_length)]
#     energy = [np.sqrt(np.mean(frame**2)) for frame in frames]
#     return np.array(energy)

# def simple_dtw(audio, phonemes):
#     duration_per_phoneme = len(audio) // (len(phonemes) * (22050 // 256))
#     durations = np.array([duration_per_phoneme] * len(phonemes))
#     return durations

# def preprocess_dataset(data_dir, output_dir, cmudict_path):
#     cmudict = load_cmudict(cmudict_path)
#     os.makedirs(output_dir, exist_ok=True)
    
#     metadata_file = os.path.join(data_dir, 'metadata.csv')
#     with open(metadata_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
    
#     for line in lines:
#         parts = line.strip().split('|')
#         audio_id, _, text = parts
#         audio_path = os.path.join(data_dir, 'wavs', f'{audio_id}.wav')
        
#         # Load and normalize audio
#         sr, audio = scipy.io.wavfile.read(audio_path)
#         if sr != 22050:
#             raise ValueError(f"Expected sample rate 22050, got {sr}")
#         audio = audio / np.max(np.abs(audio))
        
#         # Extract features
#         phonemes = simple_g2p(text, cmudict)
#         durations = simple_dtw(audio, phonemes)
#         pitch = extract_pitch(audio)
#         energy = extract_energy(audio)
        
#         # Save preprocessed data
#         np.save(os.path.join(output_dir, f'{audio_id}_phonemes.npy'), phonemes)
#         np.save(os.path.join(output_dir, f'{audio_id}_waveform.npy'), audio)
#         np.save(os.path.join(output_dir, f'{audio_id}_durations.npy'), durations)
#         np.save(os.path.join(output_dir, f'{audio_id}_pitch.npy'), pitch)
#         np.save(os.path.join(output_dir, f'{audio_id}_energy.npy'), energy)

# if __name__ == "__main__":
#     preprocess_dataset('data/LJSpeech-1.1', 'data/preprocessed', 'cmudict-0.7b')