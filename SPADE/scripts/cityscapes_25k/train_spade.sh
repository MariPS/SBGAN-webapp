#!/bin/bash


#cityscapes_full
	#res:256
python train.py --name cityscapes_25k --dataset_mode cityscapes_full_weighted --dataroot ../SBGAN/datasets --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 32 --tf_log  --niter 30 --niter_decay 30 --no_instance \
 --checkpoints_dir weights --not_sort

	# res:128
# python train.py --name cityscapes_25k_128 --dataset_mode cityscapes_full_weighted --dataroot ../SBGAN/datasets --gpu_ids 6,7 --batchSize 32 --tf_log  --niter 15 --niter_decay 15 --no_instance \
#  --checkpoints_dir weights --not_sort  --load_size 256 --crop_size 256 --no_html
