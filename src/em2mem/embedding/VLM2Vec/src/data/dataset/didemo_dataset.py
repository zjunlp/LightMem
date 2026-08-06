import os
import sys

from datasets import load_dataset
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.data.utils.vision_utils import save_frames, load_frames, sample_frames
from src.data.utils.dataset_utils import sample_dataset
from src.model.processor import process_input_text


TASK_INST_QRY = "Find a video that includes the following described scenes:"
TASK_INST_TGT = "Understand the content of the provided video."

VIDEO_EXTENSIONS = [
    ".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm",
    ".mpg", ".mpeg", ".3gp", ".m4v", ".3g2", ".mts"
]

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    model_backbone = kwargs['model_backbone']

    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for video_path, caption in zip(batch_dict['video'], batch_dict['caption']):
        query_texts.append(process_input_text(TASK_INST_QRY, model_backbone, text=caption))
        query_images.append(None)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_path = os.path.join(video_root, os.path.basename(video_path))
        frame_dir = os.path.join(frame_root, video_name)
        if not (os.path.exists(video_path)):
            curr_extend = "." + video_path.split(".")[-1]
            for ext in VIDEO_EXTENSIONS:
                tmp_path = video_path.replace(curr_extend, ext)
                if (os.path.exists(tmp_path)):
                    video_path = tmp_path
                    break
        try:
            save_frames(video_path=video_path,
                        frame_dir=frame_dir,
                        max_frames_saved=max_frames_saved)
        except Exception as e:
            query_texts[-1] = None
            pos_texts.append(None)
            pos_images.append(None)
            neg_texts.append(None)
            neg_images.append(None)
            # assert False
            continue
        video_frame_paths = load_frames(frame_dir)
        video_frame_paths = sample_frames(video_frame_paths, num_segments=num_frames)

        pos_texts.append(process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True))
        pos_images.append({"bytes": [None] * num_frames, "paths": video_frame_paths,
                             "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames})
        neg_texts.append(None)
        neg_images.append(None)

    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images}

def keep_valid(example):
    # Only keep sample if 'pos_text' is not None
    return example["pos_text"] is not None


DATASET_PARSER_NAME = "didemo"
DATASET_HF_PATH = "friedrichor/DiDeMo"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_didemo_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset = load_dataset(DATASET_HF_PATH, split="train")
    dataset = sample_dataset(dataset, **kwargs)
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution


    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch = False)
    dataset = dataset.filter(keep_valid)                        # drop na
    dataset = dataset.select_columns(["query_text", "query_image", "pos_text", "pos_image", "neg_text", "neg_image", "global_dataset_name"])

    dataset = dataset.cast(MULTIMODAL_FEATURES)
    setattr(dataset, "num_rows", num_rows)

    return dataset
