import random

from datasets import load_dataset
from PIL import Image
from datasets.features.image import image_to_bytes
import io

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import VLM_IMAGE_TOKENS


DOCMATIX_QUERY_PROMPT = "Given the question below, retrieve the most relevant PDF document that can help answer it:\n: "
DOCMATIX_DOC_PROMPT = "Represent the following PDF pages: "
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']
    batch_size = len(batch_dict['texts'])
    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for chats, images in zip(batch_dict['texts'], batch_dict['images']):
        chat = random.choice(chats)
        query = f'{DOCMATIX_QUERY_PROMPT} Q: {chat["user"]}'
        query_texts.append(query)
        query_images.append(None)
        pos_images.append({"bytes": [i['bytes'] for i in images], "paths": [i['path'] for i in images], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(images)})
        pos_texts.append(f"{DOCMATIX_DOC_PROMPT} {''.join([VLM_IMAGE_TOKENS[model_backbone]] * len(images))}")
        neg_texts.append(None)
        neg_images.append(None)
    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images}


DATASET_PARSER_NAME = "docmatix"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_docmatix_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", None)
    assert "config_name" in kwargs, "config_name must be provided for `docmatix`, ['images', 'pdf', 'zero-shot-exp']"
    config_name = kwargs.get("config_name")
    dataset_path = kwargs.get("dataset_path", None)

    if dataset_name:
        dataset = load_dataset(dataset_name, config_name, split="train")
    elif dataset_path:
        dataset = load_dataset("parquet", data_files=dataset_path, split="train")

    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'
    # dataset = dataset.shuffle(buffer_size=8192, seed=training_args.seed)
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=128,
                          remove_columns=['images', 'texts'],
                          drop_last_batch = True)
    # dataset = dataset._resolve_features()
    # features = _infer_features_from_batch(dataset._head()) # not working: {ArrowInvalid}ArrowInvalid('Could not convert <PIL.Image.Image image mode=RGB size=128x128 at 0x7F7C794E9BD0> with type Image: did not recognize Python value type when inferring an Arrow data type')
    dataset = dataset.cast(MULTIMODAL_FEATURES)
    setattr(dataset, 'num_rows', num_rows)
    # print_master(f"Loaded {DATASET_PARSER_NAME}/{dataset_name} dataset with {num_rows} samples")
    return dataset
