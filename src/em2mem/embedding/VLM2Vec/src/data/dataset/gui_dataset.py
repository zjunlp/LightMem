from typing import List
from datasets import load_dataset
from PIL import Image
import os, ast
import torch

from torch.jit import isinstance
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils import print_master, print_rank


def process_multi_images(image_basedir, image_paths) -> List[str]:
    if not image_paths:
        return [None]

    if isinstance(image_paths, str):
        try:
            image_paths = ast.literal_eval(image_paths)
        except (ValueError, SyntaxError):
            image_paths = [image_paths]

    img_path_list = []
    for image_path in image_paths:
        img_path_list.append(os.path.join(image_basedir, image_path))

    return img_path_list


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_dir = kwargs['image_dir']
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']

    batch_size = len(batch_dict['qry_text'])
    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path in \
        zip(batch_dict['qry_text'], batch_dict['qry_image_path'],
            batch_dict['pos_text'], batch_dict['pos_image_path'],
            batch_dict.get('neg_text', [''] * batch_size), batch_dict.get('neg_image_path', [None] * batch_size)):
        if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
            print("empty inputs")
            continue
        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]) if neg_text else ''
        query_texts.append(qry_text)
        pos_texts.append(pos_text)
        neg_texts.append(neg_text)
        # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
        qry_img_paths = process_multi_images(image_dir, qry_image_path)
        pos_img_paths = process_multi_images(image_dir, pos_image_path)
        neg_img_paths = process_multi_images(image_dir, neg_image_path)
        qry_image = {"bytes": [None] * len(qry_img_paths), "paths": qry_img_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_img_paths)}
        pos_image = {"bytes": [None] * len(pos_img_paths), "paths": pos_img_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(pos_img_paths)}
        neg_image = {"bytes": [None] * len(neg_img_paths), "paths": neg_img_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(neg_img_paths)}
        query_images.append(qry_image)
        pos_images.append(pos_image)
        neg_images.append(neg_image)
    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images}


DATASET_PARSER_NAME = "gui"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_gui_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    dataset_split = kwargs.get("dataset_split", "limit_10")
    dataset = load_dataset(dataset_name, subset_name, split=f"{dataset_split}")
    column_names = dataset.column_names
    num_rows = dataset.num_rows
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{subset_name}'
    # dataset = dataset.shuffle(buffer_size=8192, seed=training_args.seed)
    remove_columns = ['qry_text', 'qry_image_path', 'pos_text', 'pos_image_path', 'qry_id', 'pos_id', 'retrieval_type']
    if 'neg_image_path' in column_names:
        remove_columns.append('neg_text')
        remove_columns.append('neg_image_path')
        remove_columns.append('neg_id')
    dataset = dataset.map(lambda x:
                          data_prepare(x, **kwargs), batched=True, batch_size=128,
                          remove_columns=remove_columns,
                          drop_last_batch = True
                          )
    # dataset = dataset._resolve_features()
    # features = _infer_features_from_batch(dataset._head()) # not working: {ArrowInvalid}ArrowInvalid('Could not convert <PIL.Image.Image image mode=RGB size=128x128 at 0x7F7C794E9BD0> with type Image: did not recognize Python value type when inferring an Arrow data type')
    dataset = dataset.cast(MULTIMODAL_FEATURES)
    setattr(dataset, 'num_rows', num_rows)
    print_master(f"Loaded {DATASET_PARSER_NAME}/{subset_name} dataset with {num_rows} samples")
    return dataset
