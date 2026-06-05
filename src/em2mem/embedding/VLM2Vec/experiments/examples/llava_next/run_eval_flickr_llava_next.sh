export PYTHONPATH=../VLM2Vec/:$PYTHONPATH


CUDA_VISIBLE_DEVICES=0 python evaluation/eval_flickr.py \
  --model_name TIGER-Lab/VLM2Vec-LLaVa-Next \
  --model_backbone llava_next \
  --max_len 256 \
  --pooling last --normalize True \
  --per_device_eval_batch_size 16 \
  --encode_output_path /home/ziyan/MMEB_eval/flickr_new/


## I -> T:
#Recall@1: 0.9400
#Recall@5: 0.9930
#Recall@10: 0.9960
#
## T -> I
#Recall@1: 0.8024
#Recall@5: 0.9494
#Recall@10: 0.9736
