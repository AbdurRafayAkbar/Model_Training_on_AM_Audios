import os, io
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

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

ds = load_dataset("AbijahKaj/telephony-amd-dataset").cast_column("audio", Audio(decode=False))

# 3 random clips per (source, label) group, plus the 6 shortest clips overall
picked = df.groupby(["source", "label"]).sample(n=3, random_state=42)
shortest = df.nsmallest(6, "duration_s")
picked = pd.concat([picked, shortest]).drop_duplicates(subset=["split", "index"])

os.makedirs("samples", exist_ok=True)
for _, r in picked.sort_values(["source", "label"]).iterrows():
    item = ds[r["split"]][int(r["index"])]
    audio, sr = sf.read(io.BytesIO(item["audio"]["bytes"]))      # decode the raw bytes ourselves
    out = f'samples/{r["source"]}__{r["label"]}__{r["split"]}{int(r["index"])}.wav'
    sf.write(out, audio, sr)
    print(f'{out:62s} {r["duration_s"]:6.2f}s  {sr} Hz')