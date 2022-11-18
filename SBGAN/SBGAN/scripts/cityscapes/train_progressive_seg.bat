@echo off

SET gpu_ids=0,1
python "SBGAN\trainers\progressive_seg_trainer.py" "--name" "cityscapes_segments" "--dataset" "cityscapes" "--save_freq" "10000" "--label_seg" "34" "--num_semantics" "35" "--epochs" "150" "--batchSize" "16"
