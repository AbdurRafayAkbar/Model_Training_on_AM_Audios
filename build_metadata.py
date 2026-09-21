import os, io, re
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

REPO = "AbijahKaj/telephony-amd-dataset"
ds = load_dataset(REPO).cast_column("audio", Audio(decode=False))
label_names = ds["train"].features["label"].names

rows = []
for split_name in ["train", "test"]:
    split = ds[split_name]
    for i in tqdm(range(len(split)), desc=split_name):
        item = split[i]
        name = item["audio"]["path"] or ""
        raw = item["audio"]["bytes"]
        info = sf.info(io.BytesIO(raw))          # reads ONLY the header
        rows.append({
            "split": split_name,
            "index": i,
            "filename": name,
            "pattern": re.sub(r"\d+", "#", name),  # digits become # to reveal the naming pattern
            "label": label_names[item["label"]],
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_s": info.duration,
            "format": info.format,
            "bytes": len(raw),
        })

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/metadata.csv", index=False)
print("Saved results/metadata.csv with", len(df), "rows")