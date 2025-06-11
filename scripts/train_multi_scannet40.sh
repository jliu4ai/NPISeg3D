#!/usr/bin/env bash

python main.py --dataset_mode=multi_obj \
               --scan_folder=../datasets/ScanNet/scans \
               --train_list=../datasets/ScanNet/train_list.json \
               --val_list=../datasets/ScanNet/val_list.json \
               --lr=5e-4 \
               --batch_size=6 \
               --epochs=600 \
               --job_name=NPISEG_HNP_batch6_epoch600_calibration_weight0.2 \
