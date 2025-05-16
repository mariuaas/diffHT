import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

from ..utils.scatter import scatter_bitpack_argmax_edges_and_values, scatter_sum_2d
from ..utils.concom import connected_components
from ..utils.segmentation import _get_token_mask, get_seg_edges
from ..utils.indexing import fast_uidx_1d, lexsort

def token_merging(
    x:Tensor, k:Tensor, seg:Tensor, amask:Tensor, num_cls_tokens:int, 
    merge_ratio:float=0.1, target:Optional[int]=None
) -> tuple[Tensor, Tensor, Tensor]:
    '''Applies Token Merging (ToMe) for superpixel transformer.

    Parameters
    ----------
    x : Tensor
        Feature tensor
    k : Tensor
        Key tensor from MSA.
    seg : Tensor
        Current segmentation tensor.
    amask : Tensor
        The attention mask for the graph.
    num_cls_tokens : int
        Number of class tokens in transformer.
    merge_ratio : float, optional
        Target ratio of merging per image. Defaults to 0.1

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        New features, segmentation, and amask after ToMe.
    '''
    B, H, W = seg.shape
    nC = num_cls_tokens
    mV = seg.view(-1).bincount()
    if target is None:
        target = int(round((H*W)**.5))

    # Get edges
    E = get_seg_edges(seg, B, H, W, True)
    nloop = torch.ne(*E)

    # Get mask of non-cls_tokens
    msk = F.pad(amask[:,nC:], (nC,0,0,0))
    nV = int(msk.sum().item())

    # Compute edge similarities
    fE = k.new_zeros(E.shape[-1])
    fE[nloop] = F.cosine_similarity(*k[msk][E[:,nloop]]).add_(1).div_(2) # type: ignore
    b_idx = torch.where(msk)[0]

    # Compute argmax and values
    amx, vals = scatter_bitpack_argmax_edges_and_values(fE, E[0], E[1], int(msk.sum().item()))

    # Lexical sort by values and batch index
    _id = lexsort(1-vals, b_idx)

    # Select floor(N*ratio) highest edges
    bc = b_idx.bincount()
    add = bc.mul(merge_ratio).floor().long()
    add = (bc - (bc - add).clip(min=target)).clip(min=0)
    start = bc.cumsum(-1) - bc
    select = (
        torch.arange(add.sum().item(), device=add.device) - 
        (add.cumsum(-1) - add).repeat_interleave(add) + 
        start.repeat_interleave(add)
    )
    u = _id[select]
    v = amx[_id][select]

    # Connected components
    loops = seg.new_ones(nV, dtype=bool) # type:ignore
    loops[u] = 0
    loops[v] = 0
    loops = torch.where(loops)[0]
    cc = connected_components(
        torch.cat([loops, u]), 
        torch.cat([loops,v]), 
        nV
    )

    # Update seg
    seg_new = cc[seg]

    # Update mask
    amask_new = F.pad(
        _get_token_mask(
            seg_new, None, B, 
            b_idx=b_idx[fast_uidx_1d(cc)]
        )[0], 
        (nC,0,0,0)
    )
    nCmask_old = F.pad(
        amask.new_zeros(1,amask.shape[-1]-nC), 
        (nC,0,0,0), 
        value=True
    ).expand(B,-1)
    nCmask_new = F.pad(
        amask_new.new_zeros(1,amask_new.shape[-1]-nC), 
        (nC,0,0,0), 
        value=True
    ).expand(B,-1)

    # Update features
    new_x = x.new_zeros(B, amask_new.shape[-1], x.shape[-1])
    new_x[nCmask_new] = x[nCmask_old]
    mV_new = scatter_sum_2d(mV, cc)
    new_x[amask_new] = scatter_sum_2d(mV.view(-1,1) * x[msk], cc) / mV_new.view(-1,1)

    # Return token merged result
    return new_x, seg_new, amask_new.add(nCmask_new)


class TokenMerging(nn.Module):

    def __init__(self, num_cls_tokens:int, merge_ratio:float=0.1, target:Optional[int]=None):
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.merge_ratio = merge_ratio
        self.target = target

    def forward(self, x:Tensor, k:Tensor, seg:Tensor, amask:Tensor):
        if self.merge_ratio <= 0:
            return x, seg, amask
        return token_merging(
            x, k, seg, amask, self.num_cls_tokens, self.merge_ratio, self.target
        )
    
    def __repr__(self):
        args = f'merge_ratio={self.merge_ratio}, active={self.merge_ratio > 0}'
        return f'{self.__class__.__name__}({args})'


