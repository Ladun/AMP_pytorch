#!/bin/bash

python main.py --train --config=configs/Humanoid.yaml --exp_name=original20240909
# python main.py --train --config=configs/Humanoid_v1.yaml --exp_name=original20240902

# python main.py --eval \
#                --experiment_path=checkpoints/Humanoid/original20240906_1 \
#                --load_postfix=best \
#                --video_path=videos/Humanoid