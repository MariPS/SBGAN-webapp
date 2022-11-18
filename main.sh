#!/bin/bash

source venv-linux/bin/activate
cd SBGAN 
streamlit run SBGAN/trainers/streamlit-app.py -- --test --name ft_ade_indoor_end2end --dataset ade_indoor --dataset_mode ade_indoor --dataroot datasets/ade_indoor --num_semantics 95 --label_seg 95 --nums_fid 433 --which_iter_D2 856795 --which_iter 852418 --which_epoch 376 --load_size 256 --crop_size 256 --checkpoints_dir weights --batchSize 1 --N 1 --no_instance
cd ..
