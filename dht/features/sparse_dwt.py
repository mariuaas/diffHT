import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from ..utils.indexing import fast_invuidx_long3d
    

def partial_dwt_sparse(
    idx:Tensor, feat:Tensor, height:int, width:int, 
    mask:Optional[Tensor]=None, coeff:float=.5
) -> tuple[Tensor, Tensor, Tensor]:
    '''Perform a partial discrete wavelet transform on sparse feature data,
    optionally using a mask to select specific features for processing.

    Parameters
    ----------
    idx : Tensor
        A 2D tensor containing the indices of features in the format [3, n] where
        the first row corresponds to x-coordinates, the second row to y-coordinates,
        and the third row to the indices along the depth or channel dimension.
    feat : Tensor
        A tensor of shape [n, m] where `n` is the number of features and `m` is the 
        number of dimensions of each feature.
    height : int
        The height of the image or space where the features are embedded.
    width : int
        The width of the image or space where the features are embedded.
    mask : Optional[Tensor]
        An optional boolean tensor of shape [n] indicating whether each feature 
        should be included (True) or ignored (False) in the transform. Defaults to None,
        where all features are processed.

    Returns
    -------
    tuple
        A tuple containing three elements:
        - idx_new: Tensor
          A tensor of new indices after the wavelet transform.
        - out_new: Tensor
          A tensor of transformed features corresponding to idx_new.
        - new_mask: Tensor
          A boolean tensor indicating which of the idx_new should be included in the
          next pass of processing.

    Notes
    -----
    This function assumes the input tensors are formatted correctly and are on the 
    same device (e.g., all on CPU or all on GPU). It is designed to handle sparse 
    data efficiently by only processing non-ignored features if a mask is provided.
    '''
    N, C = feat.shape

    # If a mask is passed, we split the indices
    if mask is not None:
        nmask = ~mask
        ignored_idx = idx[:,nmask]
        ignored_feat = feat[nmask]
        idx = idx[:,mask]
        feat = feat[mask]
    
    # Calculate x direction: Positive samples
    tgt = idx[2] // 2
    uidx, inv = fast_invuidx_long3d(torch.stack([idx[0], idx[1], tgt], 0))
    out_pos_x = feat.new_zeros(len(uidx), C)
    out_pos_x.scatter_add_(0, inv.view(-1,1).expand(-1,C), feat * coeff)
    idx_pos_x = torch.stack([idx[0][uidx], idx[1][uidx], tgt[uidx]])
    
    # Negative samples
    tgt = (idx[2] // 2) + (width // 2)
    sgn = 1 - 2 * (idx[2] % 2)
    uidx, inv = fast_invuidx_long3d(torch.stack([idx[0], idx[1], tgt], 0))
    out_neg_x = feat.new_zeros(len(uidx), C)
    out_neg_x.scatter_add_(0, inv.view(-1,1).expand(-1,C), sgn.view(-1, 1) * feat * coeff)
    idx_neg_x = torch.stack([idx[0][uidx], idx[1][uidx], tgt[uidx]])    
    
    # Concatenate positive and negative
    idx_x = torch.cat([idx_pos_x, idx_neg_x], -1)
    out_x = torch.cat([out_pos_x, out_neg_x], 0)    
    
    # Calculate y direction: Positive samples
    tgt = idx_x[1] // 2
    uidx3, inv3 = fast_invuidx_long3d(torch.stack([idx_x[0], tgt, idx_x[2]], 0))
    out_pos_y = feat.new_zeros(len(uidx3), C)
    out_pos_y.scatter_add_(0, inv3.view(-1,1).expand(-1,C), out_x * coeff)
    idx_pos_y = torch.stack([idx_x[0][uidx3], tgt[uidx3], idx_x[2][uidx3]])
    
    # Negative samples
    tgt = (idx_x[1] // 2) + (height // 2)
    sgn = 1 - 2 * (idx_x[1] % 2)
    uidx3, inv3 = fast_invuidx_long3d(torch.stack([idx_x[0], tgt, idx_x[2]], 0))
    out_neg_y = feat.new_zeros(len(uidx3), C)
    out_neg_y.scatter_add_(0, inv3.view(-1,1).expand(-1,C), sgn.view(-1, 1) * out_x * coeff)
    idx_neg_y = torch.stack([idx_x[0][uidx3], tgt[uidx3], idx_x[2][uidx3]])
    
    idx_new = torch.cat([idx_pos_y, idx_neg_y], -1)
    out_new = torch.cat([out_pos_y, out_neg_y], 0)
    
    # If mask is passed, combine the ignored indices
    if mask is not None:
        idx_new = torch.cat([idx_new, ignored_idx], -1)
        out_new = torch.cat([out_new, ignored_feat], 0)
    
    # Compute mask for the indices that should be included in the next pass
    new_mask = (idx_new[1] < (height // 2)) & (idx_new[2] < (width // 2))
    
    return idx_new, out_new, new_mask


def multiscale_dwt_sparse(
    idx:Tensor, feat:Tensor, height:int, width:int, levels:int, 
    mode:str='mean', suplvl:Optional[Tensor]=None
) -> tuple[Tensor, Tensor]:
    '''Perform a multiscale discrete wavelet transform on sparse features.

    Parameters
    ----------
    idx : Tensor
        A 2D tensor containing the indices of features in the format [3, n].
    feat : Tensor
        A tensor of shape [n, m] where `n` is the number of features and `m` is the 
        number of dimensions of each feature.
    height : int
        The original height of the image or space where the features are embedded.
    width : int
        The original width of the image or space where the features are embedded.
    levels : int
        The number of levels of wavelet transform to apply. Each level reduces the 
        spatial dimensions by a factor of two.
    mode : str
        The mode to use, either mean or ortho. Corresponds to the scaling of each level.
    suplvl : Tensor, optional
        A tensor of maximum levels corresponding to the first dimension of `idx`.
        Used to constrain the wavelet transform for each region to a fixed number of levels.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing two tensors:
        - idx: Tensor
          A tensor of indices of the sparse wavelet transform.
        - feat: Tensor
          A tensor of transformed wavelet coefficients.
    '''
    coeff = dict(
        ortho=.5**.5,
        mean=.5,
    ).get(mode, .5)
    mask = None
    for l in range(levels):
        if suplvl is not None:
            if mask is None:
                mask = (suplvl >= l)[idx[0]]
            else:
                mask = mask & ((suplvl >= l)[idx[0]])
        h, w = height // (2**l), width // (2**l)
        idx, feat, mask = partial_dwt_sparse(idx, feat, h, w, mask, coeff)
    return idx, feat


def coords2lvl(y:Tensor, x:Tensor, H:int, W:int, maxlvl:int) -> Tensor:
    '''Calculate level in the DWT hierarchy that (y, x) belongs to.

    Parameters
    ----------
    y : Tensor
        The y-coordinates.
    x : Tensor
        The x-coordinates.
    H : int
        The original height of the space or image.
    W : int
        The original width of the space or image.
    maxlvl : int
        The maximum level of the hierarchy.

    Returns
    -------
    torch.Tensor
        A tensor containing the level in the multiscale hierarchy that the given (y, x) coordinates corresponds to.
    '''
    return torch.log2(torch.minimum(H / y, W / x)).clip(0, maxlvl).floor().long()


def coords2quadlvl(y:Tensor, x:Tensor, H:int, W:int, maxlvl:int) -> tuple[Tensor, Tensor]:
    '''Calculate level in the DWT hierarchy that (y, x) belongs to and determine the quadrant.
    
    Parameters
    ----------
    y : torch.Tensor
        The y-coordinates.
    x : torch.Tensor
        The x-coordinates.
    H : int
        The original height of the space or image.
    W : int
        The original width of the space or image.
    maxlvl : int
        The maximum level of the hierarchy.
    
    Returns
    -------
    tuple of torch.Tensor
        - First tensor contains the level in the multiscale hierarchy for each coordinate.
        - Second tensor contains the quadrant index (0 for top-left, 1 for top-right, 2 for bottom-left, 3 for bottom-right).
    '''
    levels = torch.log2(torch.minimum((H-1) / y, (W-1) / x)).clip(0, maxlvl-1).floor().long()
    scale = 2**levels
    quadrants = 2*(y*scale*2 >= H).long()
    quadrants += (x*scale*2 >= W).long()
    return levels, quadrants


class DWT(nn.Module):

    '''The Discrete Haar Wavelet Transform'''
    
    def __init__(self, input_size, mode='mean'):
        super().__init__()
        self.input_size = input_size
        assert mode in ['ortho', 'mean', 'meaninv', 'doubleinv', 'quadinv']
        self.mode = mode
        self.rsqrt2 = .5**.5
            
    def forward(self, x):
        curx = x[...,:self.input_size]
        resx = x[...,self.input_size:]
        a, b = curx[...,0::2], curx[...,1::2]
        if self.mode == 'ortho':
            s = self.rsqrt2 * (a + b)
            d = self.rsqrt2 * (a - b)
        elif self.mode == 'mean':
            s = (a + b) / 2
            d = (a - b) / 2
        elif self.mode == 'meaninv':
            s = a + b
            d = a - b
        elif self.mode == 'doubleinv':
            s = 2*(a + b)
            d = 2*(a - b)
        else:
            s = 4*(a + b)
            d = 4*(a - b)
        out = torch.cat([s, d], dim=-1)
        return torch.cat([out, resx], dim=-1)
    
    def inv(self, x):
        curx = x[...,:self.input_size]
        resx = x[...,self.input_size:]
        s, d = curx[...,:self.input_size//2], curx[...,self.input_size//2:]
        out = torch.zeros_like(curx)
        if self.mode == 'ortho':
            out[...,0::2] = self.rsqrt2 * (s + d)
            out[...,1::2] = self.rsqrt2 * (s - d)
        elif self.mode == 'mean':
            out[...,0::2] = s + d
            out[...,1::2] = s - d
        elif self.mode == 'meaninv':
            out[...,0::2] = (s + d) / 2
            out[...,1::2] = (s - d) / 2
        elif self.mode == 'doubleinv':
            out[...,0::2] = (s + d) / 4
            out[...,1::2] = (s - d) / 4
        else:
            out[...,0::2] = (s + d) / 8
            out[...,1::2] = (s - d) / 8
        return torch.cat([out, resx], dim=-1)
    
class DWT2D(nn.Module):
    
    def __init__(self, input_size, mode='mean'):
        super().__init__()
        self.dwt1d = DWT(input_size, mode)
        
    def forward(self, x):
        return self.dwt1d(self.dwt1d(x.mT).mT)
    
    def inv(self, x):
        return self.dwt1d.inv(self.dwt1d.inv(x.mT).mT)


class MultiscaleDWT2(nn.Module):
    
    def __init__(self, *image_sizes, mode='mean'):
        super().__init__()
        self.sizes = image_sizes
        self.dwts = nn.ModuleList([DWT2D(s, mode) for s in image_sizes])
        
    def forward(self, x):
        x = x.clone()
        for mod in self.dwts:
            modsize = mod.dwt1d.input_size
            x[...,:modsize,:modsize] = mod(x[...,:modsize,:modsize])
        return x
    
    def inv(self, x):
        x = x.clone()
        for mod in reversed(self.dwts):
            modsize = mod.dwt1d.input_size
            x[...,:modsize,:modsize] = mod.inv(x[...,:modsize,:modsize])
        return x
    