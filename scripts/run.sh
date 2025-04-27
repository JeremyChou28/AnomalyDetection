export CUDA_VISIBLE_DEVICES=0


# Scratch
nohup python -u main.py \
  --dataset AD \
  --win_size 84 \
  --input_c 1 \
  --output_c 1 \
  --d_model 32 \
  --d_ff 64 \
  --n_heads 4 \
  --e_layers 4 \
  --dropout 0.1 \
  --num_epochs 100 \
  --batch_size 1024 \
  --mode scratch \
  --data_path ./processed_dataset > logs/AD_scratch.log 2>&1 &

# # training
# nohup python -u main.py \
#   --anormly_ratio 0.5 \
#   --num_epochs 100 \
#   --batch_size 256 \
#   --mode train \
#   --dataset AD \
#   --data_path ./processed_dataset \
#   --input_c 1 > logs/AD_train.log 2>&1 &

# wait  

# # testing
# nohup python -u main.py \
#   --anormly_ratio 0.5 \
#   --num_epochs 100 \
#   --batch_size 256 \
#   --mode test \
#   --dataset AD \
#   --data_path ./processed_dataset \
#   --input_c 1 > logs/AD_test.log 2>&1 &
  