from src.model import MMEBModel
from src.arguments import ModelArguments
from src.utils import load_processor

import torch
from transformers import HfArgumentParser, AutoProcessor
from PIL import Image
import numpy as np


model_args = ModelArguments(
    model_name='TIGER-Lab/VLM2Vec-LLaVa-Next',
    pooling='last',
    normalize=True,
    model_backbone='llava_next')

processor = load_processor(model_args)

model = MMEBModel.load(model_args, is_trainable=False)
model.eval()
model = model.to('cuda', dtype=torch.bfloat16)

# Image + Text -> Text
inputs = processor(text='<image> Represent the given image with the following question: What is in the image',
                   images=Image.open('figures/example.jpg'),
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = 'A cat and a dog'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.4414]], device='cuda:0', dtype=torch.bfloat16)

string = 'A cat and a tiger'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a tiger = tensor([[0.3555]], device='cuda:0', dtype=torch.bfloat16)
