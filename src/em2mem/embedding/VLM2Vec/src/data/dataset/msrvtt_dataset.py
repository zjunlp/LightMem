import os
import sys

from datasets import load_dataset
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.data.utils.vision_utils import save_frames, load_frames, sample_frames
from src.data.utils.dataset_utils import sample_dataset
from src.model.processor import process_input_text

TASK_INST_QRY = "Find a video that contains the following visual content:"
TASK_INST_TGT = "Understand the content of the provided video."
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):

    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    model_backbone = kwargs['model_backbone']

    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for video_name, video_path, caption in (
            zip(batch_dict['video_id'], batch_dict['video'], batch_dict['caption'])):

        query_texts.append(process_input_text(TASK_INST_QRY, model_backbone, text=caption[0]))
        query_images.append(None)

        video_path = os.path.join(video_root, video_path)
        frame_dir = os.path.join(frame_root, video_name)
        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = load_frames(frame_dir)
        video_frame_paths = sample_frames(video_frame_paths, num_segments=num_frames)

        pos_texts.append(process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True))
        pos_images.append({"bytes": [None] * num_frames, "paths": video_frame_paths,
                            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames})
        neg_images.append(None)
        neg_texts.append(None)
    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images}


DATASET_PARSER_NAME = "msrvtt"
DATASET_HF_PATH = "friedrichor/MSR-VTT"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_msrvtt_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset = load_dataset(DATASET_HF_PATH, "train_9k", split="train")
    dataset = sample_dataset(dataset, **kwargs)
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution


    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch = False)
    dataset = dataset.select_columns(["query_text", "query_image", "pos_text", "pos_image", "neg_text", "neg_image", "global_dataset_name"])

    dataset = dataset.cast(MULTIMODAL_FEATURES)
    setattr(dataset, "num_rows", num_rows)

    return dataset
