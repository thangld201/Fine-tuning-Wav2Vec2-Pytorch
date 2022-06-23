#!/bin/bash
python3 /home/thangld/no_trainer_wav2vec2/src/main.py --base /raid/data/youtube_data/1649235555-cleaned   --train_split 0.99     --max_duration 10.0    --min_duration 2.0  --max_workers 32      --chunksize 128   --checkpoint_folder ./checkpoint\  --logging_percent_per_epoch 0.01  --lr 3e-5   --weight_decay 1e-4  --batch_size 64  --epoch 1
