import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

df = pd.read_csv("results/metadata.csv")

print("=== Rows per label and split ===")
print(pd.crosstab(df["label"], df["split"], margins=True))

print("\n=== Sample rates (Hz) ===")
print(df["sample_rate"].value_counts())

print("\n=== Channels ===")
print(df["channels"].value_counts())

print("\n=== File formats ===")
print(df["format"].value_counts())

print("\n=== Duration (seconds) by label ===")
print(df.groupby("label")["duration_s"].describe().round(2))

print("\n=== Filename patterns (digits replaced by #) ===")
print(df["pattern"].value_counts().head(30))

print("\nDuplicate filenames:", df["filename"].duplicated().sum())