import os

# Settings must be placed BEFORE importing datasets
os.environ["HF_HOME"] = r"D:\hf_cache"          # where downloads are stored
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   # be patient on a slow connection
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"

from datasets import load_dataset, Audio

REPO = "AbijahKaj/telephony-amd-dataset"

ds = load_dataset(REPO)          # downloads the files, then loads them
print(ds)                        # shows splits and number of rows

# Give us raw file bytes instead of auto-decoding (no TorchCodec needed)
ds = ds.cast_column("audio", Audio(decode=False))

train = ds["train"]
print("\ncolumn_names :", train.column_names)
print("features     :", train.features)
print("label names  :", train.features["label"].names)

# Look at ONE row, but do not print the raw bytes (they are huge)
row = train[0]
print("\nkeys in row 0       :", list(row.keys()))
print("audio keys in row 0 :", list(row["audio"].keys()))
print("audio path          :", row["audio"]["path"])
audio_bytes = row["audio"]["bytes"]
print("audio bytes length  :", len(audio_bytes) if audio_bytes else None)
print("label (number)      :", row["label"])
print("label (name)        :", train.features["label"].int2str(row["label"]))