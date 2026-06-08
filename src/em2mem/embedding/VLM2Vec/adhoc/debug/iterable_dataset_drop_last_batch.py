from datasets import Dataset
from datasets import interleave_datasets
from torch.utils.data import DataLoader

def convert_to_str(batch, dataset_name):
    batch['a'] = [f"{dataset_name}-{e}" for e in batch['a']]
    return batch

def gen1():
    for ii in range(1, 25):
        yield {"a": ii}

def gen2():
    for ii in range(1, 25):
        yield {"a": ii}

# https://github.com/huggingface/datasets/issues/6565
if __name__ == '__main__':
    dataset1 = Dataset.from_generator(gen1).to_iterable_dataset(num_shards=2)
    dataset2 = Dataset.from_generator(gen2).to_iterable_dataset(num_shards=2)
    dataset1 = dataset1.map(lambda x: convert_to_str(x, dataset_name="a"), batched=True, batch_size=10, drop_last_batch=True)
    dataset2 = dataset2.map(lambda x: convert_to_str(x, dataset_name="b"), batched=True, batch_size=10, drop_last_batch=True)

    interleaved = interleave_datasets([dataset1, dataset2], stopping_strategy="all_exhausted")

    print(f"num_workers=0")
    loader = DataLoader(interleaved, batch_size=5, num_workers=0)
    i = 0
    for b in loader:
        print(i, b['a'])
        i += 1

    print('=-' * 20)
    print(f"num_workers=1")
    loader = DataLoader(interleaved, batch_size=5, num_workers=1)
    i = 0
    for b in loader:
        print(i, b['a'])
        i += 1

    print('=-' * 20)
    print(f"num_workers=2")
    loader = DataLoader(interleaved, batch_size=5, num_workers=2)
    i = 0
    for b in loader:
        print(i, b['a'])
        i += 1

    print('=-' * 20)
    print(f"num_workers=3")
    loader = DataLoader(interleaved, batch_size=5, num_workers=3)
    i = 0
    for b in loader:
        print(i, b['a'])
        i += 1