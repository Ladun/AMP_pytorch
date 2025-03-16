# AMP_pytorch
Adversarial Model Prior pytorch implementation using Unity ML-agent environment.

In this project, we implement the reinforcement learning algorithm ourselves, and we use Unity solely for the environment.

# Prepare data

In this project, we use the [DeepMimic](https://github.com/xbpeng/DeepMimic) data for training AMP

DeepMimic motion data can be obtained from GitHub, and the path is [All Data](https://github.com/xbpeng/DeepMimic/tree/master/data).

# Training

``` 
python main.py --train --config=configs/Humanoid.yaml
```

## resume
```
python main.py --train --experiment_path=<experiment_path> --load_postfix=<best or timesteps000000>
```

# Evaluate 

```
python main.py --eval \
               --experiment_path=checkpoints/Humanoid/<experiment_name> \
               --load_postfix=best \
               --video_path=<path_to_save_video>
```