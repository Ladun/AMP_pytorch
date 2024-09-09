# AMP_pytorch
Adversarial Model Prior pytorch implementation using pybullet


# Prepare data

In this project, we use the [CMU motion capture](http://mocap.cs.cmu.edu/) data for training AMP
## Download

```
python tools/download_motion_data.py
```
## Test

```
python -m tools.mocap_player_for_mujoco --asf_file=data/asf/02.asf --amc_file=data/walk/02_01.amc
```

# Training

``` 
python main.py --train --config=configs/Humanoid.yaml
```

# Evaluate 

```
python main.py --eval \
               --experiment_path=checkpoints/Humanoid/<experiment_name> \
               --load_postfix=best \
               --video_path=<path_to_save_video>
```