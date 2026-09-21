from huggingface_hub import HfApi

repo_id = "AbijahKaj/telephony-amd-dataset"

api = HfApi()
# files_metadata=True asks the Hub to include each file's size
info = api.dataset_info(repo_id, files_metadata=True)

total_bytes = 0
for f in info.siblings:                 # "siblings" = the files inside the repo
    size = f.size or 0                  # some entries may have no size, treat as 0
    total_bytes += size
    print(f"{size / 1024**2:10.1f} MB   {f.rfilename}")

print(f"\nNumber of files : {len(info.siblings)}")
print(f"Total size      : {total_bytes / 1024**3:.2f} GB")