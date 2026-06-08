import logging

import PIL
from transformers.image_utils import ChannelDimension

logger = logging.getLogger(__name__)

import torch
import numpy as np
from ..utils import print_master

from .vlm_backbone.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from .vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from .vlm_backbone.qwen2_vl_tokenselection.modeling_qwen2_vl import \
    Qwen2VLForConditionalGeneration as Qwen2VLTokenSelectionForConditionalGeneration
from .vlm_backbone.qwen2_vl_tokenselection.processing_qwen2_vl import \
    Qwen2VLProcessor as Qwen2VLTokenSelectionProcessor

QWEN2_VL = 'qwen2_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl_tokenselection'
MODEL2BACKBONE = {  # keys are from hf_config.model_type or manually added if not provided
    'qwen2_vl': QWEN2_VL,
    'qwen2_vl_tokenselection': QWEN2_VL,
    'qwen2_vl_tokenselection': QWEN2_VL_TOKENSELECTION,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    QWEN2_VL: "<|image_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|image_pad|>",
}

VLM_VIDEO_TOKENS = {
    QWEN2_VL: "<|video_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|video_pad|>",
}

backbone2model = {
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_VL_TOKENSELECTION: Qwen2VLTokenSelectionForConditionalGeneration,
}


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    Note: due to this change, https://github.com/huggingface/transformers/commit/9215cc62d4366072aacafa4e44028c1ca187167b#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356L1102
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')
    if model_args.model_backbone in [QWEN2_VL]:
        from .vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from .vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from .vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer, size=size
        )
    elif model_args.model_backbone == QWEN2_VL_TOKENSELECTION:
        from .vlm_backbone.qwen2_vl_tokenselection.processing_qwen2_vl import Qwen2VLProcessor
        from .vlm_backbone.qwen2_vl_tokenselection.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from .vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.do_resize = data_args.resize_use_processor
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]

    # 1. iterate each pair and process, since processors do not support processing for mixed batch (contains data w/ and w/o visual inputs)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            try:
                if vlm_image_token in text:
                    if isinstance(images, PIL.Image.Image):
                        # images is a single image
                        images = [images]
                    for iid, image in enumerate(images):
                        # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                        if image.size[0] < 28 or image.size[1] < 28:
                            image = image.resize((56, 56))
                            images[iid] = image
                    inputs = processor(text=[text], images=images, return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
                elif vlm_video_token in text:
                    # TODO: check text/video data validity
                    inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
                else:
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")
            except Exception as e:
                for i in images:
                    print(i.filename)
                raise e
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(), 
        'texts': texts,
        'images': visual_inputs,
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    return inputs


def Qwen2_VL_TokenSelection_process_fn(model_inputs: dict, processor: Qwen2VLTokenSelectionProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    patch_pos, select_mask = [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            patch_pos.append(None)
            select_mask.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            image_exists = True
            # TODO only
            # handling multi-image data from videos, cannot deal with mixed image + video data
            if VLM_IMAGE_TOKENS[QWEN2_VL] in text:
                inputs = processor(text=[text], images=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            elif VLM_VIDEO_TOKENS[QWEN2_VL] in text:
                assert len(images) > 1, f"Video data must have more than 1 frame, got {len(images)}"
                inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError(f"Unsupported visual token in text: {text}")
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
                if 'patch_pos' in inputs:
                    patch_pos.append(inputs['patch_pos'])
                if 'select_mask' in inputs:
                    select_mask.append(inputs['select_mask'])
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                patch_pos.append(None)
                select_mask.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']

    if image_exists:
        if patch_pos:
            patch_pos_shape_for_padding = list(v.shape for v in patch_pos if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else (torch.zeros(patch_pos_shape_for_padding) - 1) for v in patch_pos]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=-1) for pos in key_tmp]
            patch_pos = torch.cat(padded_key, dim=0)
        if select_mask:
            select_mask_shape_for_padding = list(v.shape for v in select_mask if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else torch.ones(select_mask_shape_for_padding).bool() for v in select_mask]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=True) for pos in key_tmp]
            select_mask = torch.cat(padded_key, dim=0)

    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long()
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw
    inputs['patch_pos'] = patch_pos
    inputs['select_mask'] = select_mask

    return inputs

def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    # Formulate input text based on text, special token and instruction.
    # TBD: Reorganize the hard-code part for baselines such as internvideo2
    prompt = instruction
    if text:
        prompt = prompt + " " + text
    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]
        prompt = video_token + " " + prompt
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        prompt = image_token + " " + prompt

    return prompt


process_vlm_inputs_fns = {
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
}
