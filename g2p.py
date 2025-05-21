import numpy as np 

def load_cmudict(file_path):
    cmudict = {}
    with open(file_path, 'r') as f:
        for line in f: 
            if line.startswith(';;;'): continue
            parts = line.strip().split()
            word = parts[0].lower()
            phonemes = parts[1:]
            cmudict[word] = phonemes
    return cmudict 

def simple_g2p(text,cmudict):
    word = text.lower()
    phonemes = cmudict.get(word)
    for word in cmudict:
        if word.startswith(text):
            phonemes.extend(cmudict[word])
        else:
            phonemes = cmudict.get(word)

    return phonemes

cmudict = load_cmudict('cmudict-0.7b') #The CMU Pronouncing Dictionary

text = "I love you"
phonemes = simple_g2p(text,cmudict) #phoneme is a unit of sound in speech
print(phonemes)
