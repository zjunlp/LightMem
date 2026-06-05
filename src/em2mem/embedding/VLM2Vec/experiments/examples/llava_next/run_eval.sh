export PYTHONPATH=../VLM2Vec/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_name TIGER-Lab/VLM2Vec-LLaVA-v1.6-LoRA \
  --image_dir DATA_DIR/MMEB_test/MMEB_Test_1K_New/images/ \
  --encode_output_path OUTPUT_DIR/MMEB_eval/VLM2Vec-Full/ \
  --pooling eos --normalize True \
  --dataset_name TIGER-Lab/MMEB-eval \
  --dataset_split test \
  --subset_name N24News ImageNet-A ImageNet-R WebQA GQA Visual7W \
  --image_resolution high \
  --per_device_eval_batch_size 64
