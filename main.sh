#!/bin/bash
# python3 src/main.py --base /content/audio_20h \
# --train_split 0.99 \
# --max_duration 10.0\
# --min_duration 2.0\
# --max_workers 4\
# --chunksize 128\
# --checkpoint_folder /content/checkpoint\
# --logging_percent_per_epoch 0.33\
# --lr 3e-5\
# --weight_decay 1e-4\
# --batch_size 32\
# --epoch 5
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --base /content/audio_5h --train_split 0.99 --max_duration 10.0 --min_duration 2.0 --max_workers 2 --chunksize 128 --checkpoint_folder /content/checkpoint --logging_percent_per_epoch 0.1 --lr 3e-5 --weight_decay 1e-4 --batch_size 16 --epoch 1