import os, io
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

pd.set_option("display.width", 200)

def source_of(row):
    p = str(row["pattern"])
    if row["format"] == "FLAC":              return "flac_batch"
    if p.startswith("response_"):            return "response"
    if p.startswith("carrier_"):             return "carrier"
    if p.startswith(("fr_", "es_", "de_")):  return "lang_prefix"
    if row["sample_rate"] == 8000:           return "hex_8k"
    return "plain_english"

df = pd.read_csv("results/metadata.csv")
df["source"] = df.apply(source_of, axis=1)

# ---------------- PART A: composition ----------------
print("=== PART A: source x split (all rows) ===")
print(pd.crosstab(df["source"], df["split"], margins=True))
print("Rows with missing source:", df["source"].isna().sum())

# ---------------- PART B: content comparison ----------------
df["n_samples"] = (df["duration_s"] * df["sample_rate"]).round().astype(int)
df = df[~df["duration_s"].between(9.99, 10.01)]        # drop clips cut at the 10 s cap
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

ds = load_dataset("AbijahKaj/telephony-amd-dataset").cast_column("audio", Audio(decode=False))

def load_audio(split, idx):
    raw = ds[split][int(idx)]["audio"]["bytes"]
    x, _ = sf.read(io.BytesIO(raw), dtype="float32")
    return x

def envelope(x, sr):
    frame = int(0.020 * sr)                 # 20 ms windows
    hop = int(0.010 * sr)                   # step 10 ms
    n_frames = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]   # sample positions of every window
    rms = np.sqrt((x[idx] ** 2).mean(axis=1))                              # loudness per window
    return np.log(rms + 1e-6)               # log: gain changes become a constant offset

def similarity(a, b):
    n = min(len(a), len(b))
    c = np.corrcoef(a[:n], b[:n])[0, 1]
    return -1.0 if np.isnan(c) else c

results = []
for _, t in tqdm(test.iterrows(), total=len(test), desc="comparing"):
    pool = train[(train["label"] == t["label"]) & (train["sample_rate"] == t["sample_rate"])]
    twins = pool[pool["n_samples"] == t["n_samples"]]
    if len(twins) == 0:
        continue
    controls = pool.drop(twins.index)               # same label, different length
    twins = twins.sample(min(10, len(twins)), random_state=0)
    controls = controls.sample(min(10, len(controls)), random_state=0)

    t_env = envelope(load_audio("test", t["index"]), t["sample_rate"])
    def best(cands):
        return max(similarity(t_env, envelope(load_audio("train", c["index"]), c["sample_rate"]))
                   for _, c in cands.iterrows())

    results.append({"filename": t["filename"], "label": t["label"],
                    "twin_best": best(twins), "control_best": best(controls)})

res = pd.DataFrame(results)
print("\n=== PART B: test clips compared:", len(res), "===")

bins = [-1.01, 0.5, 0.7, 0.9, 0.97, 1.01]
names = ["<0.5", "0.5-0.7", "0.7-0.9", "0.9-0.97", ">=0.97"]
for col in ["twin_best", "control_best"]:
    print(f"\nBest correlation, {col}:")
    print(pd.cut(res[col], bins=bins, labels=names).value_counts().reindex(names))

print("\nMedian best correlation by label:")
print(res.groupby("label")[["twin_best", "control_best"]].median().round(3))
print("\nTop 10 most similar pairs:")
print(res.sort_values("twin_best", ascending=False).head(10).round(3))