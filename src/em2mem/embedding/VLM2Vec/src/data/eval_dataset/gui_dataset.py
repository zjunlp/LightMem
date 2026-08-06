from typing import List
from datasets import load_dataset
import os, ast

from torch.jit import isinstance
from src.data.dataset.gui_dataset import DATASET_PARSER_NAME
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS


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
    if 'qry_text' in batch_dict:
        text_keyname = 'qry_text'
        image_keyname = 'qry_image_path'
    else:
        assert 'cand_text' in batch_dict, "either 'qry_text' or 'cand_text' must be in the batch_dict"
        text_keyname = 'cand_text'
        image_keyname = 'cand_image_path'
    
    batch_size = len(batch_dict[text_keyname])

    query_texts, query_images, dataset_infos = [], [], []
    for qry_text, qry_image_path, cand_id, retrieval_type in \
        zip(batch_dict[text_keyname], batch_dict[image_keyname], batch_dict['cand_id'], batch_dict.get('retrieval_type', [None] * batch_size)):
        if (not qry_text and not qry_image_path):
            print("empty inputs")
            continue
        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        query_texts.append([qry_text])
        # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
        qry_img_paths = process_multi_images(image_dir, qry_image_path)
        qry_image = {"bytes": [None] * len(qry_img_paths), "paths": qry_img_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_img_paths)}
        query_images.append([qry_image])
        dataset_infos.append({
            "cand_id": cand_id,
            "retrieval_type": retrieval_type,
        })

    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    if text_keyname == 'qry_text':
        return {"query_text": query_texts, "query_image": query_images, "dataset_infos": dataset_infos}
    else:
        return {"cand_text": query_texts, "cand_image": query_images, "dataset_infos": dataset_infos}


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_gui_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    dataset_split = kwargs.get("dataset_split", "limit_10_ood_test")
    dataset = load_dataset(dataset_name, subset_name, split=f"{dataset_split}")
    column_names = dataset.column_names
    num_sample_per_subset = kwargs.get("num_sample_per_subset", None)
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    dataset = dataset.to_iterable_dataset(num_shards=1)
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{subset_name}'
    # dataset = dataset.shuffle(buffer_size=8192, seed=training_args.seed)
    available_columns = ['qry_text', 'qry_image_path', 'qry_id', 'cand_text', 'cand_image_path', 'cand_id', 'retrieval_type']
    remove_columns = [col for col in available_columns if col in column_names]
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          remove_columns=remove_columns, drop_last_batch = False)
    return dataset
