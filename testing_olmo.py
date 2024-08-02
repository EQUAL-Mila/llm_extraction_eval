import numpy as np
from cached_path import cached_path

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
import math

from transformers import GPTNeoXForCausalLM, AutoTokenizer



# Update these paths to what you want:
data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy")
data_order_file_path = "/network/scratch/y/yash.more/llm_extraction_eval/global_indices.npy"

train_config_path = "./config_local.yaml"

print(".. here before loading")
cfg = TrainConfig.load(train_config_path)
print(".. here after loading")
print(type(cfg))

print(".. building mmap")
dataset = build_memmap_dataset(cfg, cfg.data)

print(".. here after building mmap")
# print(dataset)
print(cfg.data)
batch_size = cfg.global_train_batch_size

print(".. loading mmap indices")
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)

print(".. here after loading mmap")
print(global_indices[:10])
print(global_indices[-10:])
"""

some_indices = global_indices[[0,1,2,3,4,5]] 

def get_batch_instances(batch_idx: int, some_indices) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
        
    batch_instances = []
    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)
    return batch_instances


# Get all 2048 x 2048 token IDs in the first batch.
print(".. getting batch instances")
l = get_batch_instances(1, some_indices)
print(l)
"""

n_samples = len(dataset)
n_batches = math.ceil(n_samples / batch_size)

print(f"Total number of batches: {n_batches}")

valid_indices = np.arange(0, len(dataset))

print('The number of valid indices is:')
print(len(valid_indices))
# Select a small sample from valid indices
some_indices = valid_indices[[0, 1, 2, 3, 4, 5]]

def get_batch_instances(batch_idx: int, indices) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = indices[batch_start:batch_end]

    batch_instances = []
    for index in batch_indices:
        if index >= len(dataset):
            print(f"Index {index} is out of bounds")
            continue  # or raise an error
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)
    return batch_instances

# Get all 2048 x 2048 token IDs in the first batch.
print(".. getting batch instances")
l = get_batch_instances(2700, valid_indices)
print(len(l[0]), print(len(l)))


# from transformers import GPTNeoXForCausalLM, AutoTokenizer

# # model = GPTNeoXForCausalLM.from_pretrained(
# #   "EleutherAI/pythia-70m-deduped",
# #   revision="step3000",
# #   cache_dir="./pythia-70m-deduped/step3000",
# # )

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

# print(".. here after loading tokenizer")
# # inputs = tokenizer("Hello, I am", return_tensors="pt")
# # tokens = model.generate(**inputs)
# decoded = tokenizer.decode(l[0])
# print(decoded)
# print("\n --- \n")

# decoded = tokenizer.decode(l[1])
# print(decoded)
# print("\n --- \n")

# decoded = tokenizer.decode(l[2])
# print(decoded)
# print("\n --- \n")

# decoded = tokenizer.decode(l[3])
# print(decoded)
# print("\n --- \n")