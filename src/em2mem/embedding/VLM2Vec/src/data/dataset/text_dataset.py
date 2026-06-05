import os
import random
from typing import Dict, List, Sized
from torch.utils.data import DataLoader, Sampler
import lightning as L
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    def __init__(
            self,
            data_name: str,
            number_training_samples: int = 1_000_000,
            neg_per_sample: int = 1,
            pos_per_sample: int = 1,
            seed: int = 777,
    ):
        super().__init__()
        data_path = DATA[data_name]['data_path']
        self.data_name = data_name
        self.instruction = DATA[data_name]['instruction']
        self.enable_cross_batch_negative_sampling = DATA[data_name].get('enable_cross_batch_negative_sampling', True)
        self.number_training_samples = number_training_samples
        self.neg_per_sample = neg_per_sample
        self.pos_per_sample = pos_per_sample
        self.seed = seed
        # print(f"Seed: {self.seed}")
        self.rng = random.Random(self.seed)

        self.data, self.cluster = self.get_data(data_name, data_path, number_training_samples)

    def get_data(self, data_name: str, data_path: str = None, number_data: int = 1_000_000):
        print_master(f"Loading data {data_name}...")
        dataset = datasets.load_dataset(data_name, split='train')

        max_num_worker_suggest = 1
        # try:
        #     max_num_worker_suggest = len(os.sched_getaffinity(0))
        # except Exception:
        #     pass

        if len(dataset) > number_data:
            cluster = set(dataset['cluster'])
            example_per_cluster = math.ceil(number_data / len(cluster))
            cluster_with_id = dataset.map(lambda example, idx: {'id': idx, 'cluster': example['cluster']},
                                          with_indices=True, num_proc=max_num_worker_suggest,
                                          remove_columns=dataset.column_names,
                                          load_from_cache_file=True)
            cluster_with_id = cluster_with_id.to_pandas()
            # group by cluster
            cluster_with_id = cluster_with_id.groupby('cluster')['id'].apply(list).reset_index()
            cluster_with_id = cluster_with_id.to_dict(orient='records')
            # sort by the number of examples in the cluster
            cluster_with_id.sort(key=lambda x: len(x['id']))

            # get the examples
            selected_index = []
            for clus in cluster_with_id:
                in_cluster_index = clus['id']
                in_cluster_index.sort()
                in_cluster_index = self.rng.sample(in_cluster_index, min(len(in_cluster_index), example_per_cluster))
                selected_index.extend(in_cluster_index)

            if len(selected_index) < number_data:
                all_data_index = list(range(len(dataset)))
                self.rng.shuffle(all_data_index)
                for idx in all_data_index:
                    if idx not in selected_index:
                        selected_index.append(idx)
                    if len(selected_index) >= number_data:
                        break
            selected_index.sort()
            dataset = dataset.select(selected_index)

        print_master(f"Assigning cluster to each example for the dataset {data_name} of size {len(dataset)}...")
        cluster = dataset.map(lambda example, idx: {'cluster': example['cluster'], 'id': idx}, with_indices=True,
                              num_proc=max_num_worker_suggest,
                              remove_columns=dataset.column_names,
                              load_from_cache_file=True)
        # group by cluster
        cluster = cluster.to_pandas()
        cluster = cluster.groupby('cluster')['id'].apply(list).reset_index()
        cluster = cluster.to_dict(orient='records')
        cluster.sort(key=lambda x: x['cluster'])
        cluster = {clus['cluster']: sorted(clus['id']) for clus in cluster}

        return dataset, cluster

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        pos = self.rng.sample(example['positive'], min(len(example['positive']), self.pos_per_sample))
        neg = self.rng.sample(example['negative'], min(len(example['negative']), self.neg_per_sample))
        assert len(
            pos) > 0, "At least one positive example per sample, got {} in idx {}. Please check the data {}".format(
            example, idx, self.data_name)
        assert len(
            neg) > 0, "At least one negative example per sample, got {} in idx {}. Please check the data {}".format(
            example, idx, self.data_name)
        return {
            'query_label': idx,
            'query': example['query'],  # str
            'positive': pos,  # list of str
            'negative': neg,  # list of str
            'instruction': self.instruction,
            'enable_cross_batch_negative_sampling': self.enable_cross_batch_negative_sampling,
        }


class RepLearningDataModule(L.LightningDataModule):
    def __init__(
            self, 
            langs: List[str], 
            use_retrieval_data_only: bool = False,
            num_workers: int = 4,
            seed: int = 777
            ):
        super().__init__()
        lang_to_data = {
            'en': EN_CROSS_BATCH if use_retrieval_data_only else EN_CROSS_BATCH + EN_NON_CROSS_BATCH,
            'ar': AR,
            'bn': BN,
            'de': DE,
            'es': ES,
            'fa': FA,
            'fi': FI,
            'fr': FR,
            'hi': HI,
            'id': ID,
            # 'ja': JA,
            'ko': KO,
            'ru': RU,
            'sw': SW,
            'te': TE,
            'th': TH,
            'vi': VI,
            'yo': YO,
            'zh': ZH,
        }

        self.data_names = []
        if langs == ['all']:
            for l in lang_to_data.keys():
                self.data_names.extend(lang_to_data[l])
        else:
            for l in langs:
                self.data_names.extend(lang_to_data[l])
        self.data_names = list(set(self.data_names))
        self.data_names.sort()
        # self.data_names = [self.data_names[0]]
        print(f"Data names: {self.data_names}")
        self.num_workers = num_workers
        self.seed = seed
    
    def connect(
            self,
            world_size: int = 1,
            global_rank: int = 0,
            tokenizer: PreTrainedTokenizer = None, 
            special_tokens_set: str = 't5',
            global_batch_size: int = 32,
            max_seq_length: int = 512,
            number_training_samples: int = 1_000_000,
            neg_per_sample: int = 1,
            pos_per_sample: int = 1,
            ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.special_tokens_set = SPECIAL_TOKENS[special_tokens_set]
        self.global_batch_size = global_batch_size
        self.max_seq_length = max_seq_length
        self.number_training_samples = number_training_samples
        self.neg_per_sample = neg_per_sample
        self.pos_per_sample = pos_per_sample
        self.batch_size = self.global_batch_size // self.world_size
        if self.global_batch_size % self.world_size != 0:
            self.global_batch_size = self.batch_size * self.world_size
            print(f"Global batch size must be divisible by world size. Setting global batch size to {self.global_batch_size}")
        if self.batch_size <= 0:
            self.batch_size = 1
            self.global_batch_size = self.world_size
            print(f"Batch size must be greater than 0. i.e. world_size must be less than or equal to global_batch_size. Setting batch size to {self.batch_size}")
    
    def set_epoch(self, epoch: int) -> None:
        self.seed = self.seed + epoch

    def prepare_data(self) -> None:
        for data_name in self.data_names:
            print(f"Loading {data_name} dataset.")
            # Download the dataset if not already downloaded
            load_dataset(data_name)

    def setup(self, stage: str='') -> None:
        train_datasets = []
        for data_name in self.data_names:
            ds = RepLearningDataset(
                data_name=data_name,
                number_training_samples=self.number_training_samples,
                neg_per_sample=self.neg_per_sample,
                pos_per_sample=self.pos_per_sample,
                seed=self.seed,
            )
            if len(ds) > 0:
                train_datasets.append(ds)
                if self.global_rank == 0:
                    print(f"Loaded {data_name} dataset with {len(ds)} samples.")
            else:
                print(f"Skipping {data_name} dataset as it has no samples.")
        assert len(train_datasets) > 0, f"No datasets loaded. Please check the data names: {self.data_names}"
        self.train_ds = ConcatRepLearningDataset(train_datasets)
    
    def train_dataloader(self) -> DataLoader:
        max_num_worker_suggest = 1
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
        num_workers = min(self.num_workers, max_num_worker_suggest)
        collator = RepLearningCollator(
            tokenizer=self.tokenizer,
            special_tokens=self.special_tokens_set,
            max_seq_length=self.max_seq_length,
            label_pad_token_id=-100
        )
        each_data_sizes = [len(dataset) for dataset in self.train_ds.modality2dataset]
        cluster_infor = [dataset.cluster for dataset in self.train_ds.modality2dataset]
        sampler = ClusteringDataSampler(
            each_data_sizes=each_data_sizes,
            cluster_info=cluster_infor,
            global_batch_size=self.global_batch_size,
            shuffle=True,
            num_replicas=self.world_size,
            rank=self.global_rank,
            seed=self.seed,
            drop_last=False,
        )

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collator,
        )


def get_dataloaders(
        fabric: L.Fabric, 
        data_module: RepLearningDataModule,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments, 
        model_args: ModelArguments,
        training_args: TrainingArguments,
        epoch: int = 0,
        is_cross_batch_loss=True
        ):
    print(f"Creating dataloaders for epoch {epoch} with cross_batch_loss={is_cross_batch_loss}")
    batch_size = training_args.global_batch_size if is_cross_batch_loss else training_args.gc_chunk_size * fabric.world_size
    print(f"Batch size: {batch_size}")
    data_module.connect(
        world_size=fabric.world_size,
        global_rank=fabric.global_rank,
        tokenizer=tokenizer, 
        special_tokens_set=model_args.universal_learner_backbone_type,
        global_batch_size=training_args.global_batch_size if is_cross_batch_loss else training_args.gc_chunk_size * fabric.world_size,
        max_seq_length=data_args.max_seq_length,
        number_training_samples=data_args.number_training_samples,
        neg_per_sample=data_args.neg_per_sample,
        pos_per_sample=data_args.pos_per_sample,
    )
    if fabric.global_rank == 0:
        data_module.prepare_data()
    fabric.barrier()
    data_module.set_epoch(epoch)
    with fabric.rank_zero_first():
        data_module.setup()
        train_dataloader = data_module.train_dataloader()
        train_dataloader = fabric.setup_dataloaders(
            train_dataloader,
            use_distributed_sampler=False,
            move_to_device=True
        )
    return train_dataloader


if __name__=='__main__':
    from transformers import AutoTokenizer
    from lightning import seed_everything

    seed_everything(777)
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')
    dm = RepLearningDataModule(langs=['en'], num_workers=0, seed=777, use_retrieval_data_only=False)
    dm.connect(
        world_size=4,
        global_rank=0,
        tokenizer=tokenizer,
        special_tokens_set='xlm-r',
        global_batch_size=64,
        max_seq_length=512,
        number_training_samples=1_000_000,
        neg_per_sample=32000,
        pos_per_sample=3000,
    )
    dm.setup()
    dl = dm.train_dataloader()
    for batch in tqdm(dl): 
        pass


