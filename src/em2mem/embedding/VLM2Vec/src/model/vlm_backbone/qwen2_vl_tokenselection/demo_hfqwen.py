import requests
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration,AutoModel,AutoProcessor


model_path="Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,).to("cuda:0")

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_path,
                            min_pixels=min_pixels, 
                            max_pixels=max_pixels
                        )
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": "Extract text from pdf"
            }
        ]
    }
]

image = Image.open(requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw)
text = processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
)
inputs = processor( text=[text], images=[image],).to("cuda:0")

# Inference: Generation of the output
generated_ids = model.generate(inputs, max_new_tokens=128)
generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)