import os
import json
import re

# Define the datasets
datasets = [
    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
    "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA",
    "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS",
    "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"

]


# Define the root directory containing the experiment directories
checkpoint_paths = [
    # v2 baselines
    # "/fsx/home/ruimeng/runs/v3vec-baseline/gme2b/mmeb/",
    # "/fsx/home/ruimeng/runs/v3vec-baseline/gme7b/mmeb/",
    # "/fsx/home/ruimeng/runs/v3vec-baseline/lamra/mmeb/",
    # "/fsx/home/ruimeng/runs/v3vec-baseline/colpali/mmeb/",

    # unified data, qwenresize
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/",
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.video.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/",
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+vidore+visrag.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/",
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/",
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora16.IB128.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/",
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora16.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/",
    #   "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.visdoc.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/",
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora16.noIB.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/",
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora16.IB32.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/",
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+video.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval/"
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora16.IB32.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval"
    # "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video_v2.qwenresize.lora16.noIB.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-4000/eval"
# "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb+video_v2+split_visdoc.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100//checkpoint-1000/eval"
#    "/fsx/home/yeliu/runs/mmeb/qwen2vl_202_2B.mmeb20+visdoc+video.qwenresize.lora32.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval"
# "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora16.IB0.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval"
# "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video_v2.qwenresize.lora16.noIB.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-5000/eval"
# "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb+video_v2+split_visdoc.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-5000/eval"
# "/fsx/home/ruimeng/runs/mmeb/qwen2vl_2B-002-6.mmeb20_vidore1_videohound2_mteb15-v2-cap100k-rerun.qwenresize.lora16.bs1024pergpu128-ib64-droplast.GCq8p8.NormTemp002.lr5e5.step5kwarm200.maxlen2k.8H100/checkpoint-4000/eval"
"/fsx/home/yeliu/runs/mmeb/qwen2vl_7B.mmeb+video_v2+split_visdoc.qwenresize.lora16.bs512pergpu64.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-3000/eval"

]


# Function to extract step number from checkpoint directory name
def extract_step(checkpoint_name):
    match = re.search(r'checkpoint-(\d+)', checkpoint_name)
    return int(match.group(1)) if match else float('inf')


# Dictionary to hold all gathered scores, organized by experiment
gathered_scores_by_exp = {}

# Loop through checkpoint directories
for checkpoint_path in checkpoint_paths:
    print(checkpoint_path)
    step = extract_step(checkpoint_path)
    experiment_dir = checkpoint_path.split("/")[-3]

    # Check if it is a checkpoint directory, and a valid checkpoint dir
    if str.isdigit(str(step)):
        # Initialize a dictionary to store scores for this checkpoint
        checkpoint_scores = {"experiment": experiment_dir, "checkpoint": str(step)}
    else:
        checkpoint_scores = {"experiment": experiment_dir, "checkpoint": "default"}

    # Go through each dataset and check if the corresponding score file exists
    for dataset in datasets:
        score_file = os.path.join(checkpoint_path, f"{dataset}_score.json")  # Score file named like DatasetName_score.json

        # Check if the score file exists
        if os.path.isfile(score_file):
            with open(score_file, "r") as f:
                score_data = json.load(f)  # Load the score JSON
                checkpoint_scores[dataset] = score_data.get("acc", "N/A")  # Assuming 'acc' is the key for accuracy
        else:
            checkpoint_scores[dataset] = "N/A"  # If no score file, set to 'N/A'
    print(checkpoint_scores)

    # Append the scores for this checkpoint to the respective experiment group
    gathered_scores_by_exp[experiment_dir] = checkpoint_scores



print('\n' * 5)
# Print gathered scores in a comma-separated format
header = ["experiment", "checkpoint"] + datasets
print(",".join(header))  # Print header

for experiment, scores in gathered_scores_by_exp.items():
    row = [scores["experiment"], scores["checkpoint"]] + [str(scores[dataset]) for dataset in datasets]
    print(",".join(row))  # Print each row of scores



header = ["dataset"] + list(gathered_scores_by_exp.keys())
print(",".join(header))  # Print header
# Additional Block: Print results per experiment, transposed (dataset per row, step per column)
# Print dataset names in the first column, and the scores for each checkpoint in subsequent columns
for dataset in datasets:
    row = []
    for experiment, scores in gathered_scores_by_exp.items():
        row.append(str(scores[dataset]))
    print(",".join([dataset] + row))  # Print header


import pandas as pd

# Collect rows
rows = []
for dataset in datasets:
    row = [dataset]
    for experiment in gathered_scores_by_exp.keys():
        row.append(gathered_scores_by_exp[experiment][dataset])
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows, columns=header)

# Save to CSV
df.to_csv("output_scores.csv", index=False)
print("CSV saved to output_scores.csv")



# header = ["dataset"] + list(gathered_scores_by_exp.keys())
# print(",".join(header))  # Print header
# # Additional Block: Print results per experiment, transposed (dataset per row, step per column)
# # Print dataset names in the first column, and the scores for each checkpoint in subsequent columns
# for dataset in datasets:
#     print(",".join([dataset, str(scores[dataset])]))
#     for experiment, scores in gathered_scores_by_exp.items():
#         print(f"\nResults for {experiment}:")
#
