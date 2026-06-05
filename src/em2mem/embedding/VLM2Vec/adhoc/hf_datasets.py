from datasets import load_dataset

# official example from https://huggingface.co/docs/datasets/en/stream
def official_example():
    dataset = load_dataset("ethz/food101", split="validation")
    dataset = dataset.to_iterable_dataset()
    dataset = dataset.shuffle(buffer_size=1024, seed=42)
    # dataset = dataset.map(add_prefix, remove_columns=["image", "label"])  # this works
    dataset = dataset.map(add_prefix, remove_columns=["image", "label"], drop_last_batch=True, batched=True, batch_size=1024)  # this also works
    # dataset = load_dataset("ethz/food101", streaming=True)
    for batch in dataset:
        print(batch)
        pass

def add_prefix(example):
    example['text'] = [f'label: {l}' for l in example['label']]
    return example

def data_prepare(batch_dict, *args, **kwargs):
    return batch_dict

def load_mmeb():
    dataset = load_dataset("TIGER-Lab/MMEB-train", "OK-VQA", split="original")
    dataset = dataset.select(range(1000))  # step 1 select (works)
    dataset = dataset.to_iterable_dataset()
    dataset = dataset.shuffle(buffer_size=1024 * 16, seed=42)  # step 2 shuffle (works)
    dataset = dataset.map(lambda x: data_prepare(x), batched=True, batch_size=1024 * 4)  # cannot use drop_last_batch=True
    # dataset = dataset._resolve_features()
    for batch in dataset:
        print(batch)
        pass



if __name__ == "__main__":
    # official_example()
    load_mmeb()