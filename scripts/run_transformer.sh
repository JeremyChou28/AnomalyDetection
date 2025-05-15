# !/bin/bash
current_datetime=$(date +'%Y-%m-%d-%H-%M-%S')
echo $current_datetime

export CUDA_VISIBLE_DEVICES=0

winsize=1
input_c=84
output_c=84
d_model=128
d_ff=1024
n_heads=8
e_layers=3
batch_size=16384
mode=scratch

# d_model_list=(256 512 1024)
# d_ff_list=(1024 2048 4096)
# n_heads_list=(2 4 8)
# e_layers_list=(1 2 3)

# for d_model in ${d_model_list[@]}; do
#   for d_ff in ${d_ff_list[@]}; do
#     for n_heads in ${n_heads_list[@]}; do
#       for e_layers in ${e_layers_list[@]}; do
#         echo "Running with d_model=$d_model, d_ff=$d_ff, n_heads=$n_heads, e_layers=$e_layers"
#         nohup python -u main_ad.py \
#           --dataset AD \
#           --win_size $winsize \
#           --input_c $input_c \
#           --output_c $output_c \
#           --d_model $d_model \
#           --d_ff $d_ff \
#           --n_heads $n_heads \
#           --e_layers $e_layers \
#           --dropout 0.1 \
#           --num_epochs 100 \
#           --batch_size $batch_size \
#           --mode $mode \
#           --data_path ./processed_dataset > logs/AD_dm${d_model}_dff${d_ff}_nh${n_heads}_el${e_layers}_${mode}_${current_datetime}.log 2>&1 &
#         wait
#       done
#     done
#   done
# done

# Scratch
nohup python -u main_ad.py \
  --dataset AD \
  --win_size $winsize \
  --input_c $input_c \
  --output_c $output_c \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --e_layers $e_layers \
  --dropout 0.1 \
  --num_epochs 100 \
  --batch_size $batch_size \
  --mode $mode \
  --data_path ./processed_dataset > logs/AD_${mode}_${current_datetime}.log 2>&1 &

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
  