# !/bin/bash
current_datetime=$(date +'%Y-%m-%d-%H-%M-%S')
echo $current_datetime

export CUDA_VISIBLE_DEVICES=0

winsize=1
input_c=84
output_c=84
d_model=128
d_channel=4
d_ff=1024
n_heads=8
e_layers=1
batch_size=16384
mode=scratch


d_channel_list=(2 4 8)
d_model_list=(128 256 512)
n_heads_list=(2 4 8)
e_layers_list=(1 2 3)

for d_model in ${d_model_list[@]}; do
  for d_channel in ${d_channel_list[@]}; do
    for n_heads in ${n_heads_list[@]}; do
      for e_layers in ${e_layers_list[@]}; do
        echo "Running with d_model=$d_model, d_channel=$d_channel, n_heads=$n_heads, e_layers=$e_layers"
        nohup python -u main_ad.py \
          --dataset AD \
          --model_name ours \
          --win_size $winsize \
          --input_c $input_c \
          --output_c $output_c \
          --d_model $d_model \
          --d_channel $d_channel \
          --d_ff $d_ff \
          --n_heads $n_heads \
          --e_layers $e_layers \
          --dropout 0.1 \
          --num_epochs 100 \
          --batch_size $batch_size \
          --mode $mode \
          --data_path ./processed_dataset > logs/AD_dm${d_model}_dff${d_ff}_nh${n_heads}_el${e_layers}_${mode}_${current_datetime}.log 2>&1 &
        wait
      done
    done
  done
done

# # Scratch
# nohup python -u main_ad.py \
#   --dataset AD \
#   --model_name ours \
#   --win_size $winsize \
#   --input_c $input_c \
#   --output_c $output_c \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --n_heads $n_heads \
#   --e_layers $e_layers \
#   --dropout 0.1 \
#   --num_epochs 100 \
#   --batch_size $batch_size \
#   --mode $mode \
#   --data_path ./processed_dataset > logs/AD_${mode}_${current_datetime}.log 2>&1 &