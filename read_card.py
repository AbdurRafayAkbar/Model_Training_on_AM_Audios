import os
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # hide the harmless warning

from huggingface_hub import hf_hub_download

REPO = "AbijahKaj/telephony-amd-dataset"

# Returns the local path; uses the cached copy since we already downloaded it
path = hf_hub_download(REPO, "README.md", repo_type="dataset")
print("Local path:", path)
print("-" * 60)
print(open(path, encoding="utf-8").read())