# You can find the script gme_inference.py in https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py
from lamra_inference import LamRAQwen2VL

texts = [
    "What kind of car is this?",
    "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023."
]
images = [
    'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Tesla_Cybertruck_damaged_window.jpg/800px-Tesla_Cybertruck_damaged_window.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/2024_Tesla_Cybertruck_Foundation_Series%2C_front_left_%28Greenwich%29.jpg/960px-2024_Tesla_Cybertruck_Foundation_Series%2C_front_left_%28Greenwich%29.jpg',
]

lamra = LamRAQwen2VL("code-kunkun/LamRA-Ret")

# Single-modal embedding
e_text = lamra.get_text_embeddings(texts=texts)
e_image = lamra.get_image_embeddings(images=images)
print((e_text * e_image).sum(-1))
## tensor([0.2281, 0.6001], dtype=torch.float16)

# How to set embedding instruction
e_query = lamra.get_text_embeddings(texts=texts, instruction='Find an image that matches the given text.')
# If is_query=False, we always use the default instruction.
e_corpus = lamra.get_image_embeddings(images=images, is_query=False)
print((e_query * e_corpus).sum(-1))
## tensor([0.2433, 0.7051], dtype=torch.float16)

# Fused-modal embedding
e_fused = lamra.get_fused_embeddings(texts=texts, images=images)
print((e_fused[0] * e_fused[1]).sum())
## tensor(0.6108, dtype=torch.float16)
