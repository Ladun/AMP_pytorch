#!/bin/bash

python main.py --train --config=configs/Humanoid.yaml --exp_name=original20240911
# python main.py --train --config=configs/Humanoid_v1.yaml --exp_name=original20240902

# python main.py --eval \
#                --experiment_path=checkpoints/Humanoid/original20240909_1 \
#                --load_postfix=best \
#                --video_path=videos/Humanoid

# python -m tools.mocap_player_for_mujoco --asf_file=data/asf/35.asf --amc_file=data/run/35_19.amc