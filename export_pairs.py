import os, io
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

df = pd.read_csv("results/metadata.csv")
df["n_samples"] = (df["duration_s"] * df["sample_rate"]).round().astype(int)
df = df[~df["duration_s"].between(9.99, 10.01)]          # drop clips cut at the 10 s cap
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

ds = load_dataset("AbijahKaj/telephony-amd-dataset").cast_column("audio", Audio(decode=False))

def load_audio(split, idx):
    raw = ds[split][int(idx)]["audio"]["bytes"]
    x, sr = sf.read(io.BytesIO(raw), dtype="float32")
    return x, sr

def envelope(x, sr):                     # same function as the previous step
    frame, hop = int(0.020 * sr), int(0.010 * sr)
    n_frames = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.log(np.sqrt((x[idx] ** 2).mean(axis=1)) + 1e-6)

def similarity(a, b):
    n = min(len(a), len(b))
    c = np.corrcoef(a[:n], b[:n])[0, 1]
    return -1.0 if np.isnan(c) else c

pairs = []                               # will hold (score, test_row, best_train_row)
for _, t in test.iterrows():
    pool = train[(train["label"] == t["label"]) & (train["sample_rate"] == t["sample_rate"])]
    twins = pool[pool["n_samples"] == t["n_samples"]]
    if len(twins) == 0:
        continue
    twins = twins.sample(min(10, len(twins)), random_state=0)
    t_x, sr = load_audio("test", t["index"])
    t_env = envelope(t_x, sr)
    best_score, best_row = -2.0, None
    for _, c in twins.iterrows():
        c_x, _ = load_audio("train", c["index"])
        s = similarity(t_env, envelope(c_x, sr))
        if s > best_score:
            best_score, best_row = s, c
    pairs.append((best_score, t, best_row))

pairs.sort(key=lambda p: p[0], reverse=True)              # most similar first
mid = [p for p in pairs if 0.5 <= p[0] < 0.7][-3:]        # 3 mediocre matches for contrast
chosen = [("high", p) for p in pairs[:5]] + [("mid", p) for p in mid]

os.makedirs("pairs", exist_ok=True)
for k, (tag, (score, t, c)) in enumerate(chosen, start=1):
    for role, split, row in [("test", "test", t), ("train", "train", c)]:
        x, sr = load_audio(split, row["index"])
        sf.write(f"pairs/{k:02d}_{tag}_{score:.2f}_{t['label']}_{role}.wav", x, sr)
    print(f"{k:02d} {tag:4s} corr={score:.3f} {t['label']:6s} test={t['filename']}  train={c['filename']}")