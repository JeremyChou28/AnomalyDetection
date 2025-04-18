export CUDA_VISIBLE_DEVICES=1

nohup python -u main.py \
  --anormly_ratio 0.5 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode train \
  --dataset AD \
  --data_path ./processed_dataset \
  --input_c 1 > logs/AD_train.log 2>&1 &
  