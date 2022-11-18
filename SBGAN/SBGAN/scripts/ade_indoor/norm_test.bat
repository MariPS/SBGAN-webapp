@echo off

SET gpu_ids=0,1
python "SBGAN\trainers\norm_test.py" "--name" "ade_indoor_segment" "--dataset" "ade_indoor"  "--label_seg" "94" "--num_semantics" "95" "--epochs" "50" "--batchSize" "16" "--save_freq" "13680"
