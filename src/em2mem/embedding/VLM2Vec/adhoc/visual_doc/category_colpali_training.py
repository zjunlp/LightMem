from datasets import load_dataset, Dataset
from collections import defaultdict
import os
from tqdm import tqdm

# Load dataset
dataset = load_dataset("vidore/colpali_train_set", split="train")

# Group by source
source_splits = defaultdict(list)
for example in tqdm(dataset):
    source_splits[example['source']].append(example)

# Output directory
output_dir = "/fsx/sfr/data/MMEB/Visual_Doc/vidore"
os.makedirs(output_dir, exist_ok=True)

# Save each split as a Parquet file
for source, examples in source_splits.items():
    print(f"{source}: {len(examples)} examples")
    file_path = os.path.join(output_dir, f"{source}.parquet")

    # Convert to HuggingFace Dataset then save as Parquet
    hf_dataset = Dataset.from_list(examples)
    hf_dataset.to_parquet(file_path)

print(f"Saved {len(source_splits)} source-based splits as Parquet to {output_dir}/")
