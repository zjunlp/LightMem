from datasets import load_dataset
from PIL import Image
from datasets.features.image import image_to_bytes
import io
import os


from torch.jit import isinstance
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import VLM_IMAGE_TOKENS


QUERY_INSTRUCTION = "Retrieve documents that best answer the following query: "
TARGET_INSTRUCTION = "Generate an embedding for the following document, in the format of images: "
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']
    image_dir = kwargs['image_dir']
    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for query, doc_id, neg_doc_ids in zip(batch_dict['query_text'], batch_dict['positive_document_ids'], batch_dict['negative_document_ids']):
        query = QUERY_INSTRUCTION + query
        query_texts.append(query)
        query_images.append(None)
        pos_texts.append(TARGET_INSTRUCTION + VLM_IMAGE_TOKENS[model_backbone])
        path = os.path.join(image_dir, f"{doc_id[0]}.png")
        pos_images.append({"bytes": [None], "paths": [path], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]})
        neg_texts.append('')
        neg_images.append(None)
    if len(query_texts) == 0:
        print('something went wrong')
    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images}


DATASET_PARSER_NAME = "pixmo"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_pixmo_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", None)
    dataset_split = kwargs.get("dataset_split", "train")
    dataset = load_dataset(dataset_name, split=dataset_split)

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
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=128,
                          remove_columns=['query_text', 'positive_document_ids'],
                          drop_last_batch = True)
    dataset = dataset.cast(MULTIMODAL_FEATURES)
    setattr(dataset, 'num_rows', num_rows)
    return dataset
