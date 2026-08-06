# VLM2Vec-V2: Unified Multimodal Embedding for Videos, Images, and Documents

<a target="_blank" href="https://arxiv.org/abs/2507.04590">
<img style="height:22pt" src="https://img.shields.io/badge/-V2 Paper%20-black?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://arxiv.org/abs/2410.05160">
<img style="height:22pt" src="https://img.shields.io/badge/-V1 Paper-black?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/TIGER-AI-Lab/VLM2Vec">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/VLM2Vec/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/MMEB-V2">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset(V2)-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/MMEB-eval">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset(V1)-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/VLM2Vec">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Leaderboard-red?style=flat"></a>
<a target="_blank" href="https://x.com/WenhuChen/status/1844577017930694984">
<img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter"></a>
<br>


This repository contains the official code and data for **VLM2Vec-V2**, a unified framework for learning powerful multimodal embeddings across diverse visual formats including images, videos, and visual documents.

Our work introduces **MMEB-V2**, a comprehensive benchmark with 78 tasks designed to systematically evaluate embedding models across these modalities. VLM2Vec-V2 sets a new state-of-the-art, outperforming strong baselines across all categories.

This is an open-source project, and we welcome contributions from the community. We are particularly interested in additions of new functionalities, support for new datasets, bug fixes, and improvements to documentation. Please feel free to open an issue to discuss your ideas or submit a pull request!

📌 Please see our CHANGELOG for the latest features and bug fixes! [![CHANGELOG](https://img.shields.io/badge/View-CHANGELOG-blue)](./CHANGELOG.md)


> ## 🚨 Major V2 Update Alert (June 2025) 🚨
>
> This repository has been updated to **V2**, which is a complete overhaul of the codebase. The previous VLM2Vec code has been archived and can be found in the `v1` branch.
>
> **Warning:** **Please back up any local work before proceeding**. If you have a local clone from before this update, you must reset your main branch to sync with the new code. 
>
> For detailed instructions, please see the "**How to Upgrade to V2**" section below.
> 
> Your feedback on this transition process is highly appreciated. If you run into any problems, please let us know by opening an issue.



---

## 🔥 News
- **[2025-05] VLM2Vec-v2 is released!** We introduce a unified embedding framework for **images, videos, and visual documents**. Our new **MMEB-V2 benchmark**, featuring 78 diverse tasks, is also available. The VLM2Vec-V2 model outperforms previous versions and strong specialized baselines.
- **[2025-01] 🎉 VLM2Vec is accepted to ICLR 2025.**
- **[2024-10]** The technical report, code, data, and model for VLM2Vec are all available online.

<details>
  <summary>📜 View Older Updates</summary>

- [2025-02] 🔥 Two new VLM2Vec models are released, based on Qwen2VL 2B and 7B, achieving 60.1 and 65.8 (new SOTA!) on the MMEB benchmark. Check them out ([2B]([url](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-2B)), [7B]([url](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-7B)))!
- [2025-02] We are starting to work on more advanced features and extensions for VLM2Vec, and will document all changes in the ```CHANGELOG.md```. If any changes conflict with previously supported features, please feel free to raise an issue here. Thank you in advance!
- [2025-01] We have updated our [training data](https://huggingface.co/datasets/TIGER-Lab/MMEB-train). Each subset now contains two splits: ```original``` and ```diverse_instruction```. The ```original``` split is provided to support the reproduction of our paper results. The ```diverse_instruction``` split includes paraphrased instructions for each task, designed to enhance instruction diversity and improve the model's robustness to unseen instructions and tasks. Moving forward, our future releases will primarily use the ```diverse_instruction``` split.
- [2024-12] We have released the [MMEB leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB). Feel free to contact us if you want to include your model.
- [2024-12] We have released a new variant of VLM2Vec built on the LLaVa-Next backbone, which is currently our best-performing version: https://huggingface.co/TIGER-Lab/VLM2Vec-LLaVa-Next.
- [2024-10] VLM2Vec has been integrated into [vLLM](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language_embedding.py).

</details>

## Key Updates
- Unified framework for training and evaluating embedding models for three modalities of data: images, videos, and visual documents.
- MMEB v2 Benchmark: Extend v1 benchmark with videos and visdoc tasks, including 78 tasks in total.
- VLM2Vec-v2.0: brand-new embedding model based on Qwen2-VL-2B.
- Easy configuration of training and evaluation using yaml files (see examples in `experiments/release`).
- Easy extension of new datasets by creating and registering customized data loader (see examples in `src/dataset/`).


## How to Upgrade to V2

1. **Back Up Your Local Changes (Critical!)** The update process will discard any uncommitted changes on your local main branch. If you have work you want to save, commit it to a new branch or use `git stash`.
2. **Reset Your Local Repository to V2.** Run the following commands to fetch the new `main` branch and reset your local copy to match it.

```bash
    # Make sure you are on your main branch first
    git checkout main
    # Fetch all recent updates from the remote and remove stale branch references
    git fetch --all --prune
    # Force your local main branch to match the new remote main branch
    git reset --hard origin/main
```


## Model
VLM2Vec-V2 fine-tunes a state-of-the-art Vision-Language Model (VLM) using instruction-guided contrastive training. The model learns to produce a single, powerful fixed-dimensional embedding for any combination of text, image, video, and document inputs.

For current V2 models, we use **Qwen2-VL** as the model backbone, which capably handles interleaved sequences of text and visuals, variable resolutions, and long-form inputs like videos and visual documents.

[//]: # (<img width="768" alt="abs" src="assets/train_vlm.png">)

### Released checkpoints
- **[VLM2Vec-v2.0 (Qwen2VL-2B)](https://huggingface.co/VLM2Vec/VLM2Vec-V2.0)**: Our primary model, demonstrating strong, balanced performance across all modalities.

<details>
<summary> V1 checkpoints </summary>

- [VLM2Vec-Qwen2VL (7B)](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-7B)
- [VLM2Vec-Qwen2VL (2B)](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-2B)
- [VLM2Vec-LLaVa-Next](https://huggingface.co/TIGER-Lab/VLM2Vec-LLaVa-Next)
- [VLM2Vec-Phi3.5V](https://huggingface.co/TIGER-Lab/VLM2Vec-Full)
</details>

 
## MMEB-v2 Benchmark
We introduce **MMEB-V2**, an expanded benchmark that includes **78 total datasets** covering images, videos, and visual documents.
- **New Video Tasks**: video retrieval, moment retrieval, video classification, and video QA.
- **New visual document task**: visual document retrieval.
- **Check out [MMEB-v2 datasets](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2)** at Huggingface.

<img width="768" alt="MMEB-V2 Overview" src="assets/mmeb_v2.png">

## Data Download
Please refer to `experiments/release_public/data/download_data.sh`.

## Training
Our training process uses a curated dataset from three main sources: video-language data (LLaVA-Hound), visual document data (Vidore, VisRAG), and image-text data (MMEB-train). We use an interleaved sub-batching strategy for stable and effective contrastive learning.

**How to run**: please see examples in `experiments/public/train`.

## Evaluation
DDP inference on multiple GPUs is supported. The whole evaluation process is streamlined and can be finished within hours. 

**How to run**: please see examples in `experiments/public/eval`. 

## Heads-up for Reproducing Baseline Models
- GME: requires an older version of the transformers library <=4.51.3.
- MomentSeeker: we recommend using a single GPU with a batch size of 10. This is due to a limitation in baseline processors that cannot handle mixed batches of image and text-only data.

## Citation
```
@article{jiang2024vlm2vec,
  title={VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks},
  author={Jiang, Ziyan and Meng, Rui and Yang, Xinyi and Yavuz, Semih and Zhou, Yingbo and Chen, Wenhu},
  journal={arXiv preprint arXiv:2410.05160},
  year={2024}
}

@article{meng2025vlm2vecv2,
  title={VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents},
  author={Rui Meng and Ziyan Jiang and Ye Liu and Mingyi Su and Xinyi Yang and Yuepeng Fu and Can Qin and Zeyuan Chen and Ran Xu and Caiming Xiong and Yingbo Zhou and Wenhu Chen and Semih Yavuz},
  journal={arXiv preprint arXiv:2507.04590},
  year={2025}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/VLM2Vec&type=Date)](https://star-history.com/#TIGER-AI-Lab/VLM2Vec&Date)

