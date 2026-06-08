from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_IMAGE_TOKENS, Qwen2_VL_process_fn
from src.utils import batch_to_device
from PIL import Image
import torch

model_args = ModelArguments(
    model_name='VLM2Vec/VLM2Vec-V2.0',
    pooling='last',
    normalize=True,
    model_backbone='qwen2_vl',
    lora=True
)
data_args = DataArguments()

processor = load_processor(model_args, data_args)
model = MMEBModel.load(model_args)
model = model.to('cuda', dtype=torch.bfloat16)
model.eval()

# Image + Text -> Text
inputs = processor(text=f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image',
                   images=Image.open('../../../assets/example.jpg'),
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
qry_output = model(qry=inputs)["qry_reps"]

string = 'A cat and a dog'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.3281]], device='cuda:0', dtype=torch.bfloat16)

string = 'A cat and a tiger'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a tiger = tensor([[0.2871]], device='cuda:0', dtype=torch.bfloat16)


# Batch processing
processor_inputs = {
    "text": [f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image',
          f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image'],
    "images": [Image.open('../../../assets/example.jpg'),
            Image.open('../../../assets/example.jpg')],
}
inputs = Qwen2_VL_process_fn(
    processor_inputs,
    processor)
inputs = batch_to_device(inputs, "cuda")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    qry_output = model(qry=inputs)["qry_reps"]

processor_inputs = {
    "text": ['A cat and a dog', 'A cat and a tiger'],
    "images": [None, None],
}
inputs = Qwen2_VL_process_fn(
    processor_inputs,
    processor)
inputs = batch_to_device(inputs, "cuda")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    tgt_output = model(tgt=inputs)["tgt_reps"]
print(model.compute_similarity(qry_output, tgt_output))
# tensor([[0.2974, 0.2390],
#         [0.2978, 0.2390]], device='cuda:0', grad_fn=<MmBackward0>)
