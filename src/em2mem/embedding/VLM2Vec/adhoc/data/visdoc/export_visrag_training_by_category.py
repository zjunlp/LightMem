from datasets import load_dataset, Dataset
from collections import defaultdict
import os
from tqdm import tqdm

# Base output directory
base_output_dir = "/mnt/disks/embedding/data/vlm2vec/MMEB-train/visdoc/visrag"

# Dataset name to subfolder mapping
datasets_to_process = {
    # 'openbmb/VisRAG-Ret-Train-Synthetic-data': 'Synthetic_data',
    'openbmb/VisRAG-Ret-Train-In-domain-data': 'Train_in_domain_data',
}

# Process each dataset
for data_name, folder_name in datasets_to_process.items():
    print(f"\nProcessing: {data_name}")
    # Create output subfolder
    output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset(data_name, split="train")
    # Group by source
    source_splits = defaultdict(list)
    for example in tqdm(dataset, desc="Categorizing"):
        source_splits[example['source']].append(example)

    # Save each split as a Parquet file
    for source, examples in source_splits.items():
        print(f"{source}: {len(examples)} examples")

        file_path = os.path.join(output_dir, f"{source}.parquet")
        hf_dataset = Dataset.from_list(examples)
        hf_dataset.to_parquet(file_path)

    print(f"Saved {len(source_splits)} source-based splits to: {output_dir}/")
