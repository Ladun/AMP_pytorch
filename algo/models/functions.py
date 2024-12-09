

import torch
from torch import nn

def init_normal_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def init_orthogonal_weights(m, gain=1):
    if isinstance(m, nn.Linear):
        orthogonal_init(m.weight)
        nn.init.constant_(m.bias, gain)
        
def init_xavier_uniform(m, gain=1):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    

def orthogonal_init(tensor, gain=1):
    '''
    https://github.com/implementation-matters/code-for-paper/blob/094994f2bfd154d565c34f5d24a7ade00e0c5bdb/src/policy_gradients/torch_utils.py#L494
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


