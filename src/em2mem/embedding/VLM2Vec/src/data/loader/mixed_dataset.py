from datasets.distributed import split_dataset_by_node

from src.data.dataset.base_pair_dataset import AutoPairDataset
from src.data.dataset.hf_datasets import interleave_datasets
from src.utils import print_master
import torch

def init_mixed_dataset(dataset_config, model_args, data_args, training_args):
    weights = [d['weight'] for d in dataset_config.values()]
    w_sum = sum(weights)
    probs = [w / w_sum for w in weights]
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    train_datasets = []
    for data_idx, (global_dataset_name, dataset_config) in enumerate(dataset_config.items()):
        train_dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, **dataset_config)
        print_master(f"\t\tDataset#{data_idx} (dataset_parser={dataset_config.get('dataset_parser', 'n/a')}): {global_dataset_name}, num_rows={train_dataset.num_rows}, prob={probs[data_idx] * 100.0}")
        train_datasets.append(train_dataset)

    if training_args.interleave_batch_size and training_args.interleave_batch_size <= 1.0:
        interleave_batch_size = training_args.per_device_train_batch_size * world_size * training_args.interleave_batch_size
    else:
        interleave_batch_size = training_args.interleave_batch_size
    total_num_rows = sum([d.num_rows for d in train_datasets])
    print_master(f"\nInitializing interleave datasets:"
                 f"\n\t\tworld_size={world_size}"
                 f"\n\t\ttotal num rows={total_num_rows}"
                 f"\n\t\tglobal batch size={training_args.per_device_train_batch_size * world_size}"
                 f"\n\t\testimated num step per epoch={total_num_rows/(training_args.per_device_train_batch_size * world_size)}"
                 f"\n\t\tinterleave_batch_size={interleave_batch_size}"
                 )
    assert total_num_rows >= (training_args.per_device_train_batch_size * world_size), \
        f"total_num_rows(={total_num_rows}) must be greater than or equal to global batch size (={training_args.per_device_train_batch_size * world_size}), since the last batch will be dropped."

    if len(train_datasets) > 1:
        train_dataset = interleave_datasets(train_datasets, probabilities=probs, batch_size=interleave_batch_size,
                                            seed=training_args.seed, stopping_strategy=training_args.interleave_stopping_strategy)
    else:
        train_dataset = train_datasets[0]
    if torch.distributed.is_initialized():
        train_dataset = split_dataset_by_node(train_dataset, rank=torch.distributed.get_rank(), world_size=world_size)

    return train_dataset

