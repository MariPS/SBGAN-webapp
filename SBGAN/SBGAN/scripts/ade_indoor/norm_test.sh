#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for training the progressive GAN model on segmentations
# By Samaneh Azadi
# ============================================================


#ADE_indoor
gpu_ids=0
CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/norm_test.py --name ade_indoor_segment \
--dataset ade_indoor --save_freq 5000 --label_seg 95 --num_semantics 95 --epochs 100 --batchSize 36 