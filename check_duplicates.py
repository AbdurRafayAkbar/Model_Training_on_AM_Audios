import os, hashlib
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
from datasets import load_dataset, Audio
from tqdm import tqdm

pd.set_option("display.width", 220)

df = pd.read_csv("results/metadata.csv")
ds = load_dataset("AbijahKaj/telephony-amd-dataset").cast_column("audio", Audio(decode=False))

# Fingerprint each file (same order as the CSV: train first, then test)
hashes = []
for split_name in ["train", "test"]:
    for item in tqdm(ds[split_name], desc=split_name):
        hashes.append(hashlib.md5(item["audio"]["bytes"]).hexdigest())
assert len(hashes) == len(df), "row count mismatch"
df["hash"] = hashes

# A) Filenames that appear more than once
dup_names = df[df["filename"].duplicated(keep=False)]
g = dup_names.groupby("filename").agg(
    distinct_audio=("hash", "nunique"),
    distinct_labels=("label", "nunique"),
    splits=("split", lambda s: "+".join(sorted(set(s)))),
)
print("Rows involved in filename duplicates:", len(dup_names))
print("Distinct duplicated filenames       :", len(g))
print("Same name, IDENTICAL audio          :", (g["distinct_audio"] == 1).sum())
print("Same name, DIFFERENT audio          :", (g["distinct_audio"] > 1).sum())
print("Same name, DIFFERENT labels         :", (g["distinct_labels"] > 1).sum())
print("\nSplits where duplicated names appear:")
print(g["splits"].value_counts())

# B) Byte-identical audio, whatever the filename
dup_audio = df[df["hash"].duplicated(keep=False)]
h = dup_audio.groupby("hash").agg(
    copies=("label", "size"),
    labels=("label", "nunique"),
    splits=("split", lambda s: "+".join(sorted(set(s)))),
)
print("\nRows with byte-identical audio to another row:", len(dup_audio))
print("Groups of identical audio                     :", len(h))
print("Groups with DIFFERENT labels                  :", (h["labels"] > 1).sum())
print("\nSplits of identical-audio groups:")
print(h["splits"].value_counts())