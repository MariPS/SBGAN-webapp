@echo off


python "test.py" "--name" "ade_indoor" "--dataset_mode" "ade_indoor" "--dataroot" "..\SBGAN\datasets\ade_indoor" "--no_instance" "--checkpoints_dir" "weights"
