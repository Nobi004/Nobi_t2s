import numpy as np 
def simple_dtw(audio,phonemes):
    # placeholder assume uniform duration
    duration_per_phoneme = len(audio) // len(phonemes) * 22050 // 256
    durations = np.array([duration_per_phoneme] *len(phonemes))
    return durations

# pitch extraction function
def extract_pitch(audio ,sr=22050,frame_length=1024,hop_length=512):
    frames = [audio[i:i+frame_length] for i in range(0,len(audio)-frame_length,hop_length)]
    f0 =[]
    for frame in frames:
        corr = np.correlate(frame,frame,mode='full')
        corr = corr[len(corr)//2:]
        peak = np.argmax(corr[1:]) +1
        f0.append(sr/peak)
    return np.array(f0)

# Energy extraction function
def extract_energy(audio,frame_length=1024,hop_length=512):
    frames = [audio[i:i+frame_length] for i in range(0,len(audio)-frame_length,hop_length)]
    energy = [np.sqrt(np.mean(frame**2)) for frame in frames]
    return np.array(energy)


