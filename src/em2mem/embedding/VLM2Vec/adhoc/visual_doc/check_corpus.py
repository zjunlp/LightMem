from datasets import load_dataset

dataset_path = "/fsx/sfr/data/MMEB/Visual_Doc/vidore/Infographic-VQA.parquet"

dataset = load_dataset("parquet", data_files={"train": dataset_path}, split="train")

print(dataset[0])
