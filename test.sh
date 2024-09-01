#!/bin/bash

python main.py --train --config=configs/Humanoid.yaml --exp_name=original

# python main.py --eval \
#                --experiment_path=checkpoints/Humanoid/original \
#                --load_postfix=best \
#                --video_path=videos/Humanoid