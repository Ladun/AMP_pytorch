from dataclasses import dataclass
import tyro
import os

import numpy as np

import torch.onnx


from algo.models.policy import Actor
from algo.utils.general import get_config, get_device


@dataclass
class Args:
    experiment_path: str
    postfix: str
    
    
if __name__ == "__main__":
    
    args = tyro.cli(Args)
    
    device = get_device("cpu")
    
    config = get_config(os.path.join(args.experiment_path, "config.yaml"))   
    policy = Actor(config, device)
    
    ckpt_path = os.path.join(args.experiment_path, "checkpoints", args.postfix)
    print(f"Load pretrained model from {ckpt_path}")

    policy.load_state_dict(torch.load(os.path.join(ckpt_path, "actor.pt")))  
    policy.eval()
    
    temp_x = torch.randn(1, config.env.state_dim, requires_grad=True)
    action, _, _ = policy(temp_x)
    
    torch.onnx.export(policy,
                      temp_x,
                      os.path.join(ckpt_path, "policy.onnx"),
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['continuous_actions', 'logprob', 'entropy'],
                      dynamic_axes={'input' : {0 : 'batch_size'},  
                                    'continuous_actions' : {0 : 'batch_size'},  
                                    'logprob' : {0 : 'batch_size'},  
                                    'entropy' : {0 : 'batch_size'}})