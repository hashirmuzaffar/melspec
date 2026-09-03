"""Generate librosa ground-truth fixtures for the Go parity tests.

Run with a Python environment that has librosa installed:
    python cmd/genfixtures/gen.py

Arrays are written frame-major (frames x coefficients), which is the
transpose of librosa's coefficient-major layout, to match the Go package.
"""
import os
import numpy as np
import librosa

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "internal", "testdata")
os.makedirs(OUT, exist_ok=True)

def save(name, a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    with open(os.path.join(OUT, name + ".txt"), "w") as f:
        f.write(f"{a.shape[0]} {a.shape[1]}\n")
        for row in a:
            f.write(" ".join("%.17g" % v for v in row) + "\n")
    print(f"{name:34} {a.shape}")

SR = 22050
rng = np.random.default_rng(20260903)
n = 8000
t = np.arange(n) / SR
# Deterministic broadband signal: a sweep plus noise, so every mel band and
# every cepstral coefficient is exercised.
y = (0.6 * np.sin(2 * np.pi * 440 * t * (1 + 3 * t))
     + 0.3 * np.sin(2 * np.pi * 1800 * t)
     + 0.1 * rng.standard_normal(n))
save("signal", y)

save("stft_mag", np.abs(librosa.stft(y, n_fft=2048, hop_length=512)).T)
save("stft_mag_reflect",
     np.abs(librosa.stft(y, n_fft=2048, hop_length=512, pad_mode="reflect")).T)
save("stft_mag_nocenter",
     np.abs(librosa.stft(y, n_fft=2048, hop_length=512, center=False)).T)

save("fb_slaney", librosa.filters.mel(sr=SR, n_fft=2048).T)
save("fb_htk", librosa.filters.mel(sr=SR, n_fft=2048, htk=True).T)
save("fb_nonorm", librosa.filters.mel(sr=SR, n_fft=2048, norm=None).T)
save("fb_40_300_8000",
     librosa.filters.mel(sr=SR, n_fft=2048, n_mels=40, fmin=300, fmax=8000).T)

save("mel_default", librosa.feature.melspectrogram(y=y, sr=SR).T)
save("mel_htk", librosa.feature.melspectrogram(y=y, sr=SR, htk=True).T)
save("mel_power1", librosa.feature.melspectrogram(y=y, sr=SR, power=1.0).T)
save("mel_40_300_8000",
     librosa.feature.melspectrogram(y=y, sr=SR, n_mels=40, fmin=300, fmax=8000).T)

save("mfcc_default", librosa.feature.mfcc(y=y, sr=SR).T)
save("mfcc_13", librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13).T)
save("mfcc_lifter22", librosa.feature.mfcc(y=y, sr=SR, lifter=22).T)

M = librosa.feature.melspectrogram(y=y, sr=SR)
save("powerdb_default", librosa.power_to_db(M).T)
print("\nlibrosa", librosa.__version__)
