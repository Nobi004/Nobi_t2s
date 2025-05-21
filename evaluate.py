model.eval()
with torch.no_grad():
    waveform,_,_,_ = model(text,is_inference=True)
    # Save the waveform using scipy.io.wavfile

    