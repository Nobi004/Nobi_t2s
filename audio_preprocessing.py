import numpy as np 
def simple_dtw(audio,phonemes):
    # placeholder assume uniform duration
    duration_per_phoneme = len(audio) // len(phonemes) * 22050 // 256
    durations = np.array([duration_per_phoneme] *len(phonemes))
    return durations

