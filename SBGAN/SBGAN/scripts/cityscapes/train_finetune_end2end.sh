#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for fine-tuning the progressive GAN+SPADE model end2end
# By Samaneh Azadi
# ============================================================

# cityscapes
gpu_ids=0,1,2,3,4,5,6,7
dataset=cityscapes
name_joint=$dataset
name=ft_${name_joint}_end2end
dataroot=datasets/cityscapes
ckpt=weights
spade_epoch=200
pro_iter=298900
D2_iter=0
lambda=10
bs=8
load_size=512
crp_size=512
num_semantics=35
label_seg=35
lr=0.00001
niter=50
niterd=50
nums_fid=500
save_freq=10
eval_freq=5

# =======================================
##COPY pretrained networks from their corresponding directories
# =======================================
if [ ! -d "${ckpt}/${name}" ]; then
    mkdir "${ckpt}/${name}"
fi


pro_gan_pretrained="${ckpt}/${name_joint}_segment" 
if [ ! -f "${ckpt}/${name}/${pro_iter}.pth" ]; then
    cp "${pro_gan_pretrained}/${pro_iter}.pth" "${ckpt}/${name}/"
fi

spade_pretrained="../SPADE/${ckpt}/${name_joint}"
if [ ! -f "${ckpt}/${name}/${spade_epoch}_net_G.pth" ]; then
    cp "${spade_pretrained}/${spade_epoch}_net_G.pth" "${ckpt}/${name}/"
    cp "${spade_pretrained}/${spade_epoch}_net_D.pth" "${ckpt}/${name}/"
fi



CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_seg_end2end_trainer.py --name ${name} --dataset ${dataset} \
   --dataset_mode ${dataset} --dataroot ${dataroot} --num_semantics ${num_semantics} --label_seg ${label_seg} --lr_pgan ${lr} \
  --load_size ${load_size} --crop_size ${crp_size} --checkpoints_dir ${ckpt}  --nums_fid ${nums_fid} \
  --batchSize ${bs} --which_iter_D2 ${D2_iter} --which_epoch ${spade_epoch}  --which_iter ${pro_iter}  --lambda_pgan ${lambda}\
  --niter ${niter} --niter_decay ${niterd} --eval_freq ${eval_freq} --save_epoch_freq ${save_freq} \
  --tf_log --no_instance --end2end --pretrain_D2 --continue_train --cont_train 
