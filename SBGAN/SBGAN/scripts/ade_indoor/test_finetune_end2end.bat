@echo off

SET gpu_ids=0
SET name=ft_ade_indoor_end2end
SET dataset=ade_indoor
SET dataroot=datasets\ade_indoor
SET ckpt=weights
SET spade_epoch=351
SET pro_iter=742993
SET D2_iter=747370
SET bs=1
SET load_size=256
SET crp_size=256
SET num_semantics=95
SET label_seg=95
SET nums_fid=433
python "SBGAN\trainers\progressive_seg_end2end_trainer.py" "--test" "--name" "%name%" "--dataset" "%dataset%" "--dataset_mode" "%dataset%" "--dataroot" "%dataroot%" "--num_semantics" "%num_semantics%" "--label_seg" "%label_seg%" "--nums_fid" "%nums_fid%" "--which_iter_D2" "%D2_iter%" "--which_iter" "%pro_iter%" "--which_epoch" "%spade_epoch%" "--load_size" "%load_size%" "--crop_size" "%crp_size%" "--checkpoints_dir" "%ckpt%" "--batchSize" "%bs%" "--N" "5" "--no_instance"