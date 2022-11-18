#!/bin/bash

#cityscapes_full
python test.py --name cityscapes_25k --dataset_mode cityscapes_full_weighted --dataroot ../SBGAN/datasets --no_instance
