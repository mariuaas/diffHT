import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import NamedTuple, Optional
from .extractor import DPXExtractorResult
from itertools import accumulate


def _interpolate_pos_embed(proj, feat_size, old_pos_size, new_pos_size):
    wdata = proj.weight.data
    S, T_old = wdata.shape
    
    if T_old - 3*feat_size**2 == old_pos_size**2:
        compute_grad = False
    elif T_old - 4*feat_size**2 == old_pos_size**2:
        compute_grad = True
    else:
        errstr = 'Inconsistency in feat_size!\n'
        errstr += f'Got T_old={T_old}, which is incompatible with:\n' 
        errstr += f'feat_size {feat_size} old_pos_size {old_pos_size}'
        raise ValueError(errstr)

    l = [0] + 3*[feat_size**2] + [old_pos_size**2]
    if compute_grad:
        l += [feat_size**2]
    l = list(accumulate(l))
    slices = [slice(a,b) for a,b in zip(l[:-1],l[1:])]
    
    gr = None
    if compute_grad:
        r,g,b,p,gr = [wdata[:,sl] for sl in slices]
    else:
        r,g,b,p = [wdata[:,sl] for sl in slices]

    old_size = (1,S,old_pos_size,old_pos_size)
    new_size = 2*[new_pos_size]
    
    pi = F.interpolate(p.view(*old_size), new_size, mode='bicubic')[0].view(S,-1)
    new_list = [r,g,b,pi] + ([gr] if gr is not None else [])    
    T_new = sum([t.shape[1] for t in new_list])
    assert T_new == (compute_grad + 3) * feat_size**2 + new_pos_size**2
    
    new_linear = nn.Linear(T_new,S).to(pi.device)
    new_linear.weight.data = torch.cat([t.view(S,-1) for t in new_list], -1)
    new_linear.bias.data = proj.bias.data    
    
    return new_linear


class LinearFreezeColumnWrapper(nn.Module):

    def __init__(self, linear_module:nn.Linear, frozen_columns:slice):
        super().__init__()
        self.register_buffer(
            'pre_weight', linear_module.weight.data[:,:frozen_columns.start].clone()
        )
        self.register_buffer(
            'post_weight', linear_module.weight.data[:,frozen_columns.stop:].clone()
        )
        self.update_weight = nn.Parameter(linear_module.weight.data[:,frozen_columns.start:frozen_columns.stop].clone())
        if linear_module.bias is not None:
            self.bias = nn.Parameter(linear_module.bias.data)
        else:
            self.register_buffer(
                'bias', linear_module.bias
            )
        self.to(self.pre_weight.data.device)

    @property
    def weight(self):
        return torch.cat([
            self.pre_weight, self.update_weight, self.post_weight
        ], -1)
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
    def as_linear(self):
        weight = self.weight
        bias = False if self.bias is None else True
        out = nn.Linear(*self.weight.mT.shape, bias=bias)
        out.weight.data = weight.data
        if bias:
            out.bias.data = self.bias.data
        return out


class DPXResult(NamedTuple):

    fV: Tensor
    seg: Tensor
    amask: Tensor


class DPXMAEResult(NamedTuple):

    fVproj: Tensor
    fVrec: Tensor
    seg: Tensor
    amask: Tensor
    dmask: Tensor
    fmask: Optional[Tensor]


class DPXEmbedder(nn.Module):

    def __init__(
        self, embed_dim, patch_size:int, 
        channels=3, compute_grad=False, num_cls_tokens=1, 
        embed_bias=True, pre_norm=True, pos_patch_size=None,
    ):
        super().__init__()
        self.pos_patch_size = pos_patch_size if pos_patch_size is not None else patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_cls_tokens = num_cls_tokens
        self.channels = channels
        self.compute_grad = compute_grad
        self.token_dim = (
            (self.channels + self.compute_grad) * self.patch_size**2 + 
            self.pos_patch_size**2
        )
        self.proj = nn.Linear(self.token_dim, embed_dim, bias=embed_bias)
        self.cls_token = nn.Parameter(
            1e-4 * torch.randn(self.num_cls_tokens, embed_dim)
        )
        self.norm_pre = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
    
    def expand_with_grad(self, eps=1e-5):
        raise NotImplementedError('Erroneous!')
        if self.compute_grad:
            return
        self.compute_grad = True
        new_token_dim = (self.channels + 1 + self.compute_grad) * self.patch_size**2
        add_embed = self.patch_size**2
        orig_weight = self.proj.weight.data
        kwargs = {"device":orig_weight.device, "dtype":orig_weight.dtype}
        add_weight = torch.randn(self.embed_dim, add_embed, **kwargs) * eps
        new_weight = torch.cat([orig_weight, add_weight], -1)
        new_proj = nn.Linear(new_token_dim, self.embed_dim, bias=self.proj.bias is not None)
        assert new_proj.weight.data.shape == new_weight.shape
        new_proj.weight.data = new_weight
        if self.proj.bias is not None:
            new_proj.bias.data = self.proj.bias.data
        self.proj = new_proj
        self.token_dim = new_token_dim
        self.compute_grad = True

    def interpolate_pos_embed(self, new_pos_patch_size):
        self.proj = _interpolate_pos_embed(
            self.proj, self.patch_size, self.pos_patch_size, new_pos_patch_size
        )
        self.pos_patch_size = new_pos_patch_size
        self.token_dim = (
            (self.channels + self.compute_grad) * self.patch_size**2 + 
            self.pos_patch_size**2
        )

    def freeze_non_pos_embed(self):
        start = 3 * self.patch_size**2
        stop = start + self.pos_patch_size**2
        assert isinstance(self.proj, nn.Linear)
        self.proj = LinearFreezeColumnWrapper(
            self.proj,
            slice(start, stop)
        )

    def mae_embed(self, res:DPXExtractorResult, n_keep:int):
        B, H, W = res.seg.shape
        T, nC = self.token_dim, self.num_cls_tokens

        # Find batch indices and slice parameters
        b_idx = res.seg.view(-1).mul(B).add(res.byx[0]).unique() % B
        bc = b_idx.bincount()
        cs = bc.cumsum(-1)
        st = cs - bc
        maxdim = bc.max() + 1

        # Do not project features yet
        fV = res.fV.view(-1, T)

        # Find slices for features, vectorize, project, and add class tokens
        slices = torch.stack([st, cs], 1)
        maxdim = bc.max() + 1
        fV = torch.cat([
            torch.cat([
                fV[start:stop], 
                fV.new_zeros(maxdim+start-stop-nC, T)
            ], 0) 
            for start, stop in slices
        ], 0)

        # Update slices
        bc = bc + nC
        cs = bc.cumsum(-1)
        slices = torch.stack([cs - bc, cs], 1)
        maxdim = maxdim + nC
        amask = torch.stack([
            torch.cat([slices.new_ones(s), slices.new_zeros(maxdim-s-1)]) # type: ignore
            for s in slices.diff(dim=-1)[:,0]
        ]).float()

        # Create drop mask
        sampled_indices = amask[:,1:].multinomial(n_keep).view(-1)
        b_indices = (
            torch.arange(B, device=sampled_indices.device)
                 .view(B,1)
                 .expand(B,n_keep)
                 .reshape(-1)
        )
        dmask = torch.zeros_like(amask, dtype=torch.bool)
        dmask[:, 0] = True
        dmask[b_indices, sampled_indices + 1] = True
        amask = amask.bool()

        # Split into features for encoding and reconstruction
        fV, fVrec, fmask = self.mae_proj_norm(fV.view(B,-1,T), dmask, amask)

        return DPXMAEResult(fV, fVrec, res.seg, amask, dmask, fmask)
    

    def mae_proj_norm(self, fV, dmask, amask):
        B, M, T = fV.shape
        nC, D = self.num_cls_tokens, self.embed_dim
        cls_expand = self.cls_token.view(1, nC, D).expand(B, nC, D)
        fVproj = self.norm_pre(
            torch.cat([
                cls_expand,
                self.proj(fV[dmask[:,1:]].view(B, -1, T))
            ], 1)
        )
        fVrec = fV[(~dmask[:,1:]) & amask[:,1:]]
        fmask = torch.ones_like(fVproj[:,:,0], dtype=torch.bool)
        return fVproj, fVrec, fmask

    
    def forward(self, res:DPXExtractorResult):
        B, H, W = res.seg.shape
        T, nC = self.token_dim, self.num_cls_tokens

        # Find batch indices and slice parameters
        b_idx = res.seg.view(-1).mul(B).add(res.byx[0]).unique() % B
        bc = b_idx.bincount()
        cs = bc.cumsum(-1)
        st = cs - bc
        maxdim = bc.max() + 1

        # Embed features
        fV = self.proj(res.fV.view(-1, T))

        # Find slices for features, vectorize, project, and add class tokens
        slices = torch.stack([st, cs], 1)
        maxdim, D = bc.max() + 1, self.embed_dim
        fV = torch.cat([
            torch.cat([
                self.cls_token, 
                fV[start:stop], 
                fV.new_zeros(maxdim+start-stop-nC, D)
            ], 0) 
            for start, stop in slices
        ], 0)

        # Update slices
        bc = bc + nC
        cs = bc.cumsum(-1)
        slices = torch.stack([cs - bc, cs], 1)
        maxdim = maxdim + nC
        amask = torch.stack([
            torch.cat([slices.new_ones(s), slices.new_zeros(maxdim-s-1)]) # type: ignore
            for s in slices.diff(dim=-1)[:,0]
        ]).bool()

        return DPXResult(self.norm_pre(fV).view(B,-1,D), res.seg, amask)


class DPXMAEDecoderEmbedder(nn.Module):

    def __init__(self, embed_dim, encoder_embed, patch_size, compute_grad=False, num_cls_tokens=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder_embed = encoder_embed
        self.patch_size = patch_size
        self.compute_grad = compute_grad
        self.num_cls_tokens = num_cls_tokens
        self.proj = nn.Linear(encoder_embed, embed_dim)
        self.pos_emb = nn.Linear(patch_size**2, embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim) * 1e-4)
        self.cls_pos_emb = nn.Parameter(torch.randn(num_cls_tokens, embed_dim) * 1e-4)


    def forward(self, x, pos, amask, dmask):
        B, M = amask.shape
        D = self.embed_dim
        amask = amask.clone()
        amask[:,0] = False
        fmask = amask & (~dmask)
        embed = pos.new_zeros(B, M, D)
        embed[fmask] = self.pos_emb(pos) + self.mask_token
        embed[dmask] = self.proj(x).view(-1, D)
        embed[:,:self.num_cls_tokens] = (
            embed[:,:self.num_cls_tokens] + self.cls_pos_emb.unsqueeze(0)
        )
        return embed, fmask
