#!/usr/bin/env bash

python eval_single_obj.py --dataset=partnet \
               --dataset_mode=single_obj \
               --crop \
               --scan_folder=../datasets/PartNet/single/crops \
               --val_list=../datasets/PartNet/single/object_ids.npy \
               --val_list_classes=../datasets/PartNet/single/object_classes.txt \
               --output_dir=results/PartNet_single \
               --checkpoint=output/2024-12-18-16-05-25/best_checkpoint.pth \
               --scene_weight=0.05 \
               --cubeedge=-1
               