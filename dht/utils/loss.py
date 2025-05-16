import torch
import torch.nn as nn
import torch.nn.functional as F

_ln2 = torch.log(2*torch.ones(1)).item()
_reductions = {
    'mean': torch.mean,
    'sum': torch.sum,
    'none': nn.Identity()
}

def logcosh_loss(pred, target, c=4.0, reduction='mean'):
    cdelta = c*(pred - target)
    num = F.softplus(2*cdelta) - cdelta - _ln2
    _reduce_fn = _reductions.get('reductions', torch.mean)
    return _reduce_fn(num / c)

class LogCoshLoss(nn.Module):

    def __init__(self, c=4.0, reduction='mean'):
        super().__init__()
        self.c = c
        self.reduction = reduction
    
    def forward(self, pred, target):
        return logcosh_loss(pred, target, c=self.c, reduction=self.reduction)
    