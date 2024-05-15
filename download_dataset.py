import os
from huggingface_hub import hf_hub_download, snapshot_download
path_to_scratch = os.environ.get("SCRATCH")

# snapshot_download(repo_id="EleutherAI/pile-standard-pythia-preshuffled", repo_type="dataset", cache_dir=path_to_scratch + "/pile-pythia")
snapshot_download(repo_id="EleutherAI/pile-deduped-pythia-preshuffled", repo_type="dataset", cache_dir=path_to_scratch + "/pile-pythia")