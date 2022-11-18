#!/bin/bash

#cityscpaes
	#res:256
python train.py --name cityscapes --dataset_mode cityscapes --dataroot ../SBGAN/datasets/cityscapes --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 24 --tf_log  --niter 100 --niter_decay 100 --no_instance \
--load_size 512 --crop_size 512 --display_winsize 128 --checkpoints_dir weights

	#res:128
# python train.py --name cityscapes_128 --dataset_mode cityscapes --dataroot ../SBGAN/datasets/cityscapes --gpu_ids 0,1 --batchSize 24 --tf_log  --niter 100 --niter_decay 100 --no_instance \
# --load_size 256 --crop_size 256 --display_winsize 128 --checkpoints_dir weights --no_html
