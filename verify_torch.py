import torch
import torchaudio

print("torch version      :", torch.__version__)
print("torchaudio version :", torchaudio.__version__)
print("CUDA available     :", torch.cuda.is_available())
print("CPU threads        :", torch.get_num_threads())

# Fake 1-second audio clip: a 440 Hz tone at 8000 Hz (telephone sample rate)
sample_rate = 8000
t = torch.arange(sample_rate) / sample_rate
wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)   # shape [1, 8000]
print("waveform shape     :", tuple(wave.shape))

# Convert the waveform into a Mel spectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate, n_fft=512, hop_length=80, n_mels=64
)
mel = mel_transform(wave)
print("mel spectrogram    :", tuple(mel.shape))