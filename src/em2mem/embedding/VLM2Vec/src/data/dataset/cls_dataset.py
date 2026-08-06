import os
import sys

from datasets import load_dataset
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, MULTIMODAL_FEATURES
from src.data.eval_dataset.video_classification_utils import VIDEOCLS_LABEL_MAPPING, DATASET_INSTRUCTION
from src.data.utils.vision_utils import save_frames, load_frames, sample_frames
from src.data.utils.dataset_utils import sample_dataset
from src.model.processor import process_input_text


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution = kwargs['image_resolution']
    num_frames = kwargs['num_frames']
    video_dir, frame_base_dir = kwargs['video_dir'], kwargs['frame_dir']
    dataset_name = kwargs['dataset_name']
    model_backbone = kwargs['model_backbone']
    max_frames_saved = kwargs['max_frames_saved']

    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for query_video_id, pos_text, video_path in zip(batch_dict['video_id'], batch_dict['pos_text'], batch_dict['video_path']):
        query_video_id = str(query_video_id)
        video_path = os.path.join(video_dir, video_path)
        video_path = fr"{video_path}" # treat it as r-string to avoid path issues
        frame_dir = os.path.join(frame_base_dir, query_video_id)
        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = load_frames(frame_dir)
        video_frame_paths = sample_frames(video_frame_paths, num_segments=num_frames)

        query_texts.append(process_input_text(DATASET_INSTRUCTION[dataset_name], model_backbone, add_video_token=True))
        query_images.append({"bytes": [None] * len(video_frame_paths), 'paths': video_frame_paths,
                              "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths)})

        pos_texts.append([pos_text])
        pos_images.append(None)
        neg_texts.append(None)
        neg_images.append(None)

    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images,}


DATASET_PARSER_NAME = "video_classification"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_video_class_dataset(model_args, data_args, training_args, *args, **kwargs):
    DATASET_HF_PATH = kwargs['data_path']
    dataset = load_dataset(DATASET_HF_PATH, split='train')
    dataset = sample_dataset(dataset, **kwargs)
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch=False)
    dataset = dataset.select_columns(["query_text", "query_image", "pos_text", "pos_image", "neg_text", "neg_image"])

    dataset = dataset.cast(MULTIMODAL_FEATURES)
    setattr(dataset, "num_rows", num_rows)

    return dataset
