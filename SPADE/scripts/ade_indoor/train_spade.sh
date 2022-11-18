#!/bin/bash


#ade_indoor
	#res:256
python train.py --name ade_indoor --dataset_mode ade_indoor --dataroot ../SBGAN/datasets/ADE_indoor --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 48 --tf_log  --niter 150 --niter_decay 150 --no_instance \
 --checkpoints_dir weights --no_html

	# res:128
# python train.py --name ade_indoor_128 --dataset_mode ade_indoor --dataroot ../SBGAN/datasets/ADE_indoor --gpu_ids 2,3 --batchSize 48 --tf_log  --niter 150 --niter_decay 150 --no_instance \
#  --checkpoints_dir weights --load_size 128 --crop_size 128 --no_html
