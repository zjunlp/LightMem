from src.model.model import MMEBModel
from src.arguments import ModelArguments
from src.model.processor import load_processor
import torch
from PIL import Image

model_args = ModelArguments(
    model_name='TIGER-Lab/VLM2Vec-Full',
    pooling='last',
    normalize=True,
    model_backbone='phi3_v',
    num_crops=16)

processor = load_processor(model_args)

model = MMEBModel.load(model_args, is_trainable=False)
model.eval()
model = model.to('cuda', dtype=torch.bfloat16)


# Image + Text -> Text
inputs = processor('<|image_1|> Represent the given image with the following question: What is in the image', [Image.open(
    'figures/example.jpg')])
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = 'A cat and a dog'
inputs = processor(string)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.3008]], device='cuda:0', dtype=torch.bfloat16)

string = 'A cat and a tiger'
inputs = processor(string)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a tiger = tensor([[0.2051]], device='cuda:0', dtype=torch.bfloat16)

# Text -> Image
inputs = processor('Find me an everyday image that matches the given caption: A cat and a dog.',)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = '<|image_1|> Represent the given image.'
inputs = processor(string, [Image.open('figures/example.jpg')])
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## <|image_1|> Represent the given image. = tensor([[0.2930]], device='cuda:0', dtype=torch.bfloat16)

inputs = processor('Find me an everyday image that matches the given caption: A cat and a tiger.',)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = '<|image_1|> Represent the given image.'
inputs = processor(string, [Image.open('figures/example.jpg')])
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## <|image_1|> Represent the given image. = tensor([[0.2012]], device='cuda:0', dtype=torch.bfloat16)
