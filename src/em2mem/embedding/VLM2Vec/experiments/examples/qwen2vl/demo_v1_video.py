from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
import torch
from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info

model_args = ModelArguments(
    model_name='Qwen/Qwen2-VL-7B-Instruct',
    checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-7B',
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

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "../../../assets/example_video.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
    videos=video_inputs,
    return_tensors="pt"
)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
qry_output = model(qry=inputs)["qry_reps"]

string = 'A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## tensor([[0.4766]], device='cuda:0', dtype=torch.bfloat16)

string = 'A person dressed in a blue jacket shovels the snow-covered pavement outside their house.'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## tensor([[0.3262]], device='cuda:0', dtype=torch.bfloat16)
