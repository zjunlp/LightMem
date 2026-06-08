import logging
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments
import torch
from qwen_vl_utils import smart_resize
from PIL import Image
from src.model.processor import LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, PHI3V, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, process_vlm_inputs_fns

from src.utils import print_rank, print_master
import io

logger = logging.getLogger(__name__)
PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        examples = {'text': [e[0] for e in examples], 'images': [e[1] for e in examples]}
        inputs = process_vlm_inputs_fns[self.model_args.model_backbone](examples,
                                                                        processor = self.processor,
                                                                        max_length = self.data_args.max_len)
        inputs['texts'] = examples['text']
        inputs['images'] = examples['images']
        inputs['image_paths'] = [i.filename if hasattr(i, 'filename') else None for i in examples['images']]
        return inputs


@dataclass
class CLIPCollator:
    data_args: DataArguments
    vis_processors: AutoProcessor
    txt_processors: AutoTokenizer

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, attention_mask = [], [], []
        image_exist, text_exist = False, False
        for example in examples:
            text, image = example
            if image is not None:
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_inputs = self.vis_processors(images=image, return_tensors="pt")
                image_exist = True
                pixel_values.append(image_inputs['pixel_values'])
            if text:
                text_exist = True
            text_inputs = self.txt_processors(text, padding=getattr(self.data_args, "padding", True), max_length=self.data_args.max_len, truncation=True, return_tensors="pt")
            input_ids.append(text_inputs["input_ids"].squeeze(0))
        if text_exist:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.txt_processors.pad_token_id
            )
            attention_mask = input_ids.ne(self.txt_processors.pad_token_id)
        if image_exist:
            pixel_values = torch.cat(pixel_values, dim=0)
        if text_exist and image_exist:
            assert input_ids.size()[0]==pixel_values.size()[0]
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }

        return inputs


@dataclass
class OpenCLIPCollator:
    data_args: DataArguments
    vis_processors: AutoProcessor
    txt_processors: AutoTokenizer

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, attention_mask = [], [], []
        image_exist, text_exist = False, False
        for example in examples:
            text, image = example
            if image is not None:
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_inputs = self.vis_processors(image).unsqueeze(0)
                image_exist = True
                pixel_values.append(image_inputs)
            if text:
                text_exist = True
            text_inputs = self.txt_processors(text)
            input_ids.append(text_inputs)
        if text_exist:
            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = input_ids.ne(0)
        if image_exist:
            pixel_values = torch.cat(pixel_values, dim=0)
        if text_exist and image_exist:
            assert input_ids.size()[0]==pixel_values.size()[0]
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }

        return inputs


@dataclass
class MultimodalEvalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    encode_side: str

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            if example is None or not example:
                text, visual_input = '  ', None
            else:
                ex_text, ex_images = example[text_keyname], example[image_keyname]
                # ex_text, ex_images could be one single pair from the query side or a list of pairs from the candidates side
                has_image = isinstance(ex_images, dict) or (isinstance(ex_images, list) and all(isinstance(item, dict) for item in ex_images))
                if has_image:
                    for text, raw_images in zip(ex_text, ex_images):
                        visual_input = []
                        assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                        num_images = len(raw_images['paths'])
                        for image_idx in range(num_images):
                            bytes = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                            path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                            image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                            if bytes is None and path is None:
                                image = None
                            elif bytes is not None:
                                # vidore, image inputs are already bytes
                                image = Image.open(io.BytesIO(bytes))
                            elif path is not None:
                                # mmeb/video datasets, lazy image loading and processing
                                image = Image.open(path)
                            else:
                                print_rank(f"\n{'=' * 50}\nsomething went wrong with a data point from {example['global_dataset_name']}, neither bytes or path is given. \n\t\tquery_text: {example['query_text']}")
                            if not self.data_args.resize_use_processor and image is not None and image_resolution:
                                image = image.resize(image_resolution)
                            if image is not None and (image_resolution is not None and self.data_args.image_decay_factor is not None):
                                assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                                assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                                # TODO: this is a hacky way to decay image resolution, need to be refactored
                                max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                                width, height = image.size
                                resized_height, resized_width = smart_resize(
                                    height,
                                    width,
                                    min_pixels=self.data_args.resize_min_pixels,
                                    max_pixels=max_pixels,
                                )
                                image = image.resize((resized_width, resized_height))
                            visual_input.append(image)
                        texts.append(text)
                        visual_inputs.append(visual_input)
                else:
                    # flatten the list in cases of multiple candidates
                    for text, visual_input in zip(ex_text, ex_images):
                        texts.append(text)
                        visual_inputs.append(visual_input)
                        pass

        inputs = {'text': texts, 'images': visual_inputs}
        return inputs


    def __call__(self, examples):
        """
        :param examples: 'query_text', 'query_image', 'cand_text', 'cand_image'
        """
        process_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        if self.encode_side == 'qry':
            assert type(examples[0]['query_text']) == list or type(examples[0]['query_image']) == list, "Expect text/image to be a list, even it only contains a single element (string, dict or None)"
            inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        else:
            assert type(examples[0]['cand_text']) == list or type(examples[0]['cand_image']) == list, "Expect text/image to be a list, even it only contains a single element (string, dict or None)"
            inputs = self._get_batch_inputs(examples, "cand_text", "cand_image")

        processed_inputs = process_fn(inputs, processor=self.processor, max_length=self.data_args.max_len)
        dataset_infos = [e["dataset_infos"] for e in examples]
        return processed_inputs, dataset_infos
