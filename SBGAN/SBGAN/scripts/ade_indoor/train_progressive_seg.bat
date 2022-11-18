@echo off

SET gpu_ids=0
python "SBGAN\trainers\progressive_seg_trainer.py" "--name" "ade_indoor_segment" "--dataset" "ade_indoor" "--label_seg" "95" "--num_semantics" "95" "--epochs" "100" "--batchSize" "16" "--save_freq" "13680"