@echo off

SET gpu_ids=0
python "SBGAN\trainers\progressive_seg_trainer.py" "--name" "ade_indoor_segment" "--dataset" "ade_indoor" "--label_seg" "95" "--num_semantics" "95" "--test" "--which_iter" "519766" "--N" "4"