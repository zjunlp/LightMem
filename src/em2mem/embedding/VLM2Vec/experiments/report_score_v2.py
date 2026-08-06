import os
import json
from datetime import datetime

# ==============================================================================
# Configuration
# ==============================================================================

# ==> Unified list of experiments to process.
# Fill in the metadata for each experiment. `None` will become `null` in the JSON.

EXPERIMENTS = [
    {
        "path": "vlm2vec_exps/VLM2Vec-Qwen2VL-V2.0-2B/",
        "metadata": {
            "model_name": "VLM2Vec-Qwen2VL-V2.0-2B",
            "model_size": "2B parameters",
            "embedding_dimension": None, # Please fill in
            "max_length_tokens": None,   # Please fill in
            "model_release_date": "2025-04-01", # Please adjust this date
            "score_source": "",           # e.g., "Self-Reported" or "TIGER-Lab"
            "url": ""                    # e.g., Paper, GitHub, or Hugging Face link
        }
    },
    {
        "path": "vlm2vec_exps/VLM2Vec-Qwen2VL-V2.1-2B/",
        "metadata": {
            "model_name": "VLM2Vec-Qwen2VL-V2.1-2B",
            "model_size": "2B parameters",
            "embedding_dimension": None, # Please fill in
            "max_length_tokens": None,   # Please fill in
            "model_release_date": "2025-05-15", # Please adjust this date
            "score_source": "",           # e.g., "Self-Reported" or "TIGER-Lab"
            "url": ""                    # e.g., Paper, GitHub, or Hugging Face link
        }
    },
]


# ==============================================================================
# TODO: Your models' metadata goes here. Please fill in the required fields.
# ==============================================================================

EXPERIMENTS = [
    {
        "path": ...,
        "metadata": {
            "model_name": ...,
            "model_backbone": ...,
            "model_size": ...,
            "embedding_dimension": ...,
            "max_length_tokens": ...,
            "model_release_date": ...,
            "data_source": "Self-Reported",
            "url": ...
        }
    },
    ...
]


# ==============================================================================
# Main Processing Logic (No changes needed below this line)
# ==============================================================================


# Define the datasets grouped by modality
modality2dataset = {
    "image": [
        "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
        "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA",
        "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS",
        "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"],
    "video": [
        "K700", "SmthSmthV2", "HMDB51", "UCF101", "Breakfast",
        "MVBench", "Video-MME", "NExTQA", "EgoSchema", "ActivityNetQA",
        "DiDeMo", "MSR-VTT", "MSVD", "VATEX", "YouCook2",
        "QVHighlight", "Charades-STA", "MomentSeeker",
    ],
    "visdoc": [
        "ViDoRe_arxivqa", "ViDoRe_docvqa", "ViDoRe_infovqa", "ViDoRe_tabfquad", "ViDoRe_tatdqa", "ViDoRe_shiftproject",
        "ViDoRe_syntheticDocQA_artificial_intelligence", "ViDoRe_syntheticDocQA_energy", "ViDoRe_syntheticDocQA_government_reports", "ViDoRe_syntheticDocQA_healthcare_industry",
        "ViDoRe_esg_reports_human_labeled_v2", "ViDoRe_biomedical_lectures_v2_multilingual", "ViDoRe_economics_reports_v2_multilingual", "ViDoRe_esg_reports_v2_multilingual",
        "VisRAG_ArxivQA", "VisRAG_ChartQA", "VisRAG_MP-DocVQA", "VisRAG_SlideVQA", "VisRAG_InfoVQA", "VisRAG_PlotQA",
        "ViDoSeek-page", "ViDoSeek-doc", "MMLongBench-page", "MMLongBench-doc"
    ]
}
modality2metric = {
    "image": "hit@1",
    "video": "hit@1",
    "visdoc": "ndcg_linear@5",
}
modalities = ["image", "video", "visdoc"] # Process in this order

for experiment in EXPERIMENTS:
    base_path = experiment['path']
    experiment_metadata = experiment['metadata']
    experiment_name_for_log = os.path.basename(base_path.strip('/'))

    current_experiment_scores = {}

    print(f"\nProcessing experiment: {experiment_name_for_log}")
    print(f"Path: {base_path}")

    for modality in modalities:
        current_experiment_scores[modality] = {}
        modality_specific_result_dir = os.path.join(base_path, modality)

        for dataset_name in modality2dataset.get(modality, []):
            current_experiment_scores[modality][dataset_name] = "FILE_N/A" # Initialize

        if not os.path.isdir(modality_specific_result_dir):
            print(f"    Directory not found: {modality_specific_result_dir}")
            for dataset_name in modality2dataset.get(modality, []):
                current_experiment_scores[modality][dataset_name] = "DIR_N/A"
            continue

        for filename in os.listdir(modality_specific_result_dir):
            if filename.endswith("_score.json"):
                score_file_path = os.path.join(modality_specific_result_dir, filename)
                dataset_name_from_file = None
                for known_dataset in modality2dataset.get(modality, []):
                    if filename == f"{known_dataset}_score.json":
                        dataset_name_from_file = known_dataset
                        break

                if dataset_name_from_file:
                    try:
                        with open(score_file_path, "r") as f:
                            score_data = json.load(f)
                            current_experiment_scores[modality][dataset_name_from_file] = score_data
                    except json.JSONDecodeError:
                        print(f"      Error decoding JSON from {score_file_path}")
                        current_experiment_scores[modality][dataset_name_from_file] = "JSON_ERROR"
                    except Exception as e:
                        print(f"      Error reading file {score_file_path}: {e}")
                        current_experiment_scores[modality][dataset_name_from_file] = "READ_ERROR"

    # --- Construct and Save the Final JSON Report ---
    final_metadata = experiment_metadata.copy()
    final_metadata['report_generated_date'] = datetime.now().isoformat()
    final_output = {
        "metadata": final_metadata,
        "metrics": current_experiment_scores
    }

    output_json_path = os.path.join(base_path, f"{final_metadata['model_name']}.json")
    try:
        with open(output_json_path, "w") as f:
            json.dump(final_output, f, indent=4)
        print(f"  Report for '{experiment_name_for_log}' saved to: {output_json_path}")
    except Exception as e:
        print(f"  Error saving JSON report for '{experiment_name_for_log}' to {output_json_path}: {e}")


    # --- Print detailed main scores per dataset for easy copy to spreadsheet ---
    print(f"\n  --- Detailed Main Scores for Spreadsheet (Experiment: {experiment_name_for_log}) ---")
    for modality in modalities:
        main_metric_key = modality2metric[modality]
        for dataset_name in modality2dataset.get(modality, []):
            score_to_print_val = "NOT_FOUND_IN_RESULTS"
            modality_data = current_experiment_scores.get(modality, {})
            score_info = modality_data.get(dataset_name)

            if isinstance(score_info, dict):
                metric_value = score_info.get(main_metric_key)
                if isinstance(metric_value, (int, float)):
                    score_to_print_val = f"{metric_value:.4f}"
                else:
                    score_to_print_val = f"METRIC_KEY_MISSING ({main_metric_key})"
            elif isinstance(score_info, str):
                score_to_print_val = score_info

            print(f"{dataset_name}\t{score_to_print_val}")
        print("")

    # --- Print average scores and missing datasets per modality ---
    print(f"\n  --- Summary for Experiment: {experiment_name_for_log} ---")
    for modality in modalities:
        if modality not in current_experiment_scores:
            print(f"    Modality '{modality.upper()}' not processed.")
            continue
        main_metric_key = modality2metric[modality]
        modality_data = current_experiment_scores[modality]
        collected_metric_values = []
        datasets_missing_score_file = []
        datasets_file_found_metric_missing = []

        for dataset_name in modality2dataset.get(modality, []):
            score_info = modality_data.get(dataset_name)
            if isinstance(score_info, dict):
                metric_value = score_info.get(main_metric_key)
                if isinstance(metric_value, (int, float)):
                    collected_metric_values.append(metric_value)
                else:
                    datasets_file_found_metric_missing.append(f"{dataset_name} (metric '{main_metric_key}' missing/invalid)")
            else:
                datasets_missing_score_file.append(f"{dataset_name} (status: {score_info if score_info else 'Not Processed'})")

        if collected_metric_values:
            average_score = sum(collected_metric_values) / len(collected_metric_values)
            print(f"      Average of {modality.upper()}\t- {main_metric_key}:\t{average_score:.4f} (from {len(collected_metric_values)} datasets)")
        else:
            print(f"      Average of {modality.upper()}\t-  {main_metric_key}:\tN/A (no valid scores found)")

        if datasets_missing_score_file:
            print(f"      Datasets with missing/errored score files:")
            for ds_status in datasets_missing_score_file: print(f"        - {ds_status}")
        if datasets_file_found_metric_missing:
            print(f"      Score files found but main metric ('{main_metric_key}') missing/invalid:")
            for ds_status in datasets_file_found_metric_missing: print(f"        - {ds_status}")


print("\nProcessing complete.")