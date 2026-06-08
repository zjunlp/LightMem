export PYTHONPATH=../VLM2Vec/:$PYTHONPATH


CUDA_VISIBLE_DEVICES=0 python evaluation/eval_flickr.py \
  --model_name TIGER-Lab/VLM2Vec-Full \
  --num_crops 4 --max_len 256 \
  --model_backbone phi3_v \
  --max_len 256 \
  --pooling last --normalize True \
  --per_device_eval_batch_size 16 \
  --encode_output_path /home/ziyan/MMEB_eval/flickr_new/
