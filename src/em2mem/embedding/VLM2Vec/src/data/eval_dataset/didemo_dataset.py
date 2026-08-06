import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text


TASK_INST_QRY = "Find a video that includes the following described scenes:"
TASK_INST_TGT = "Understand the content of the provided video."

@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    model_backbone = kwargs['model_backbone']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_path, caption in zip(batch_dict['video'], batch_dict['caption']):

        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=caption)])
        query_images.append([None])

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_path = os.path.join(video_root, os.path.basename(video_path))
        frame_dir = os.path.join(frame_root, video_name)
        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)])
        cand_images.append([ImageVideoInstance(
            bytes=[None] * num_frames,
            paths=video_frame_paths,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames,
        ).to_dict()])
        dataset_infos.append({
            "cand_names": [video_name],
            "label_name": video_name,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "didemo"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_didemo_dataset(model_args, data_args, **kwargs):
    dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
    corpus = None  # No additional corpus

    return dataset, corpus
