export PYTHONPATH=../VLM2Vec/:$PYTHONPATH


CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_name TIGER-Lab/VLM2Vec-Full \
  --image_dir DATA_DIR/MMEB_test/MMEB_Test_1K_New/images/ \
  --encode_output_path OUTPUT_DIR/MMEB_eval/VLM2Vec-Full/ \
  --num_crops 4 \
  --pooling last --normalize True \
  --per_device_eval_batch_size 64 \
  --dataset_name TIGER-Lab/MMEB-eval \
  --subset_name N24News ImageNet-A ImageNet-R WebQA --dataset_split test \
