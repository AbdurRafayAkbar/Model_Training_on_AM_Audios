import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

df = pd.read_csv("results/metadata.csv")

# Group filenames into batches using their naming pattern
plain_english = {"voicemail_#.wav", "am_#.wav", "ivr_#.wav",
                 "human_real_#.wav", "human_tts_#.wav"}

def batch_of(p):
    if p in plain_english:                    return "plain_english"
    if p.startswith(("fr_", "es_", "de_")):   return "lang_prefix"
    if p.startswith("carrier_"):              return "carrier"
    if p.startswith("response_"):             return "response"
    return "other"

df["batch"] = df["pattern"].apply(batch_of)

print("Distinct filename patterns:", df["pattern"].nunique())

print("\n=== Batch x label ===")
print(pd.crosstab(df["batch"], df["label"], margins=True))

print("\n=== Sample rate x label ===")
print(pd.crosstab(df["sample_rate"], df["label"], margins=True))

print("\n=== Sample rate x batch ===")
print(pd.crosstab(df["sample_rate"], df["batch"], margins=True))

print("\n=== File format x batch ===")
print(pd.crosstab(df["format"], df["batch"], margins=True))
for fmt, names in df.groupby("format")["filename"]:
    print(fmt, "example names:", names.head(3).tolist())

print("\n=== Median duration (s) by batch and label ===")
print(df.pivot_table(index="batch", columns="label",
                     values="duration_s", aggfunc="median").round(2))

near10 = df[df["duration_s"].between(9.99, 10.01)]
print("\n=== Clips lasting 9.99-10.01 s ===")
print(near10["label"].value_counts(), "\nTotal:", len(near10))

cols = ["split", "filename", "label", "duration_s", "sample_rate"]
print("\n=== 8 shortest clips ===")
print(df.nsmallest(8, "duration_s")[cols])
print("\n=== 8 longest clips ===")
print(df.nlargest(8, "duration_s")[cols])

print("\n=== Top 25 patterns in the 'other' batch ===")
print(df[df["batch"] == "other"]["pattern"].value_counts().head(25))