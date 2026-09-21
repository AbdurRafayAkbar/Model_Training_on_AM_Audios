import pandas as pd

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

# Exact length in samples (same text + same voice = same number of samples)
df["n_samples"] = (df["duration_s"] * df["sample_rate"]).round().astype(int)

# Skip clips cut at the 10-second cap: they all share one length by design
df = df[~df["duration_s"].between(9.99, 10.01)]

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

same_label = train.groupby(["n_samples", "label"]).size().rename("n_same_label").reset_index()
any_label = train.groupby("n_samples").size().rename("n_any").reset_index()

t = test.merge(same_label, on=["n_samples", "label"], how="left")
t = t.merge(any_label, on="n_samples", how="left")
t[["n_same_label", "n_any"]] = t[["n_same_label", "n_any"]].fillna(0)

t["match_same"] = t["n_same_label"] > 0                        # twin with SAME label
t["match_other"] = (t["n_any"] - t["n_same_label"]) > 0        # twin with a DIFFERENT label (control)

print("Test clips checked:", len(t))
print("\n=== By label: share of test clips with an exact-length twin in train ===")
print(t.groupby("label").agg(clips=("filename", "size"),
      same_label_match=("match_same", "mean"),
      other_label_match=("match_other", "mean")).round(3))

print("\n=== By source ===")
print(t.groupby("source").agg(clips=("filename", "size"),
      same_label_match=("match_same", "mean"),
      other_label_match=("match_other", "mean")).round(3))

print("\nOverall: same-label =", round(t["match_same"].mean(), 3),
      "| other-label =", round(t["match_other"].mean(), 3))

print("\n=== Examples ===")
for _, r in t[t["match_same"]].head(10).iterrows():
    twins = train[(train["n_samples"] == r["n_samples"]) & (train["label"] == r["label"])]
    print(f'{str(r["filename"])[:38]:38s} {r["label"]:18s} {r["duration_s"]:6.3f}s -> '
          f'{len(twins)} train twin(s), e.g. {twins["filename"].iloc[0]}')