#!/bin/bash

# Specify the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Set the model name
model_name=S_Mamba

# Run the Python script with the appropriate arguments
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/drone_traj/ \
  --data_path drone_flight_data.txt \
  --model_id drone_traj_exp_20_10 \
  --model $model_name \
  --data DroneTraj \
  --features M \
  --seq_len 20 \
  --pred_len 10 \
  --label_len 10 \
  --e_layers 2 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Drone trajectory experiment' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 2 \
  --train_epochs 5 \
  --learning_rate 0.0001 \
  --itr 1
