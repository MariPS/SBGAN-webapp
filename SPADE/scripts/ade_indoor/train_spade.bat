@echo off

python "train.py" "--name" "ade_indoor" "--dataset_mode" "ade_indoor" "--dataroot" "..\SBGAN\datasets\ade_indoor" "--gpu_ids" "0" "--label_nc" "95" "--batchSize" "4" "--niter" "150" "--niter_decay" "150" "--no_instance" "--checkpoints_dir" "weights" "--no_html" "--save_latest_freq" "30000"