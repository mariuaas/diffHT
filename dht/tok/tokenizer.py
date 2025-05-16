import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import NamedTuple, Optional

from .state import (
    TokenizerState, GridDimensions, 
    update_tokenizer_state, finalize_tokenization, token_drop_merge
)
from ..utils.segmentation import init_graph_topology, get_seg_edges
from ..utils.scatter import scatter_mean_2d
from ..utils.imgproc import image_gradient
from ..utils.indexing import fast_uidx_long2d
from ..nn.cnn import BilateralNet, SimpleConv, HighBoost, GradOp, Arcsinh, SimpleDownsample


class TokenizerResult(NamedTuple):
    '''NamedTuple holding tokenizer features and segmentation.

    Attributes
    ----------
    fV : Tensor
        Float tensor, pixel features of shape [BHW, C].
    seg : Tensor
        Segmentation tensor, mapping to original pixel locations.
    byx: Tensor
        Image coordinates.
    bb : Tensor
        Bounding box coordinates of regions.
    nV : int
        Number of vertices, regions.
    grad : Tensor, optional
        A tensor of image gradients.
    '''
    fV: Tensor
    seg: Tensor
    byx: Tensor
    bb: Tensor
    nV: int
    grad: Optional[Tensor] = None


class DPXTokenizer(nn.Module):

    _cnn_backbones = ['bilateral', 'simple', 'downsample']

    def __init__(
        self, in_ch, hid_ch, 
        similarity='gaussian', criterion='aicc',
        iota=1, eps=1e-8, cmp=0.1, center=.0, 
        min_avg_region=6, compute_grad=False,
        unnorm_grad=True, use_varproj=True,
        cnn_backbone='downsample',
    ):
        super().__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.similarity = similarity
        self.criterion = criterion
        self.compute_grad = compute_grad
        self.unnorm_grad = unnorm_grad
        self.iota = iota
        self.eps = eps
        self.cmp = cmp
        self.center = center
        self.min_avg_region = min_avg_region
        self.cnn_backbone = (
            cnn_backbone if cnn_backbone in self._cnn_backbones 
            else 'downsample'
        )
        if self.cnn_backbone == 'bilateral':
            self.conv = BilateralNet(in_ch, hid_ch)
            self.linear = nn.Linear(hid_ch, in_ch)
            self.res = nn.Conv2d(in_ch, hid_ch, 1)
            if compute_grad:
                self.gradact = Arcsinh(2, 0, learn_lmbda=True)
                self.gradient = GradOp(k=2.0, learnable=False)
            else:
                self.gradact = nn.Identity()
                self.gradient = nn.Identity()
        elif self.cnn_backbone == 'downsample':
            self.conv = SimpleDownsample(in_ch, hid_ch)
            self.linear = nn.Linear(hid_ch, in_ch)
            if compute_grad:
                self.gradact = Arcsinh(2, 0, learn_lmbda=True)
                self.gradient = GradOp(k=2.0, learnable=False)
            else:
                self.gradact = nn.Identity()
                self.gradient = nn.Identity()
        else:
            self.conv = SimpleConv(hid_ch)
            self.linear = nn.Linear(hid_ch, in_ch)
            self.hiboost = HighBoost(k=2.0, learnable=True)
            self.gradient = GradOp(k=2.0, learnable=False)
            self.gradact = Arcsinh(2, 0, learn_lmbda=True)
            self.act = Arcsinh(3, 1e-5)
        self.varproj = None
        if use_varproj:
            self.varproj = nn.Linear(hid_ch, 1)
            self.varproj.weight.data = torch.randn(1, hid_ch) * 1e-4
            self.varproj.bias.data = -torch.ones(1)*2*torch.pi
    
    def preproc(self, img) -> tuple[Tensor, Optional[Tensor]]:
        if self.cnn_backbone == 'bilateral':
            if self.compute_grad:
                grad = self.gradact(self.gradient(img))
                img = (self.res(img) + self.conv(img))
                return img.permute(0,2,3,1).reshape(-1, self.hid_ch), grad
            img = (self.res(img) + self.conv(img))
            return img.permute(0,2,3,1).reshape(-1, self.hid_ch), None

        elif self.cnn_backbone == 'downsample':
            if self.compute_grad:
                grad = self.gradact(self.gradient(img))
                return self.conv(img).permute(0,2,3,1).reshape(-1, self.hid_ch), grad
            return self.conv(img).permute(0,2,3,1).reshape(-1, self.hid_ch), None
            
        grad = self.gradact(self.gradient(img))
        fV = self.conv(torch.cat([self.act(self.hiboost(img)), grad], 1)).permute(0,2,3,1)
        if self.compute_grad:
            return fV.reshape(-1, self.hid_ch), grad
        return fV.reshape(-1, self.hid_ch), None
    
    def expand_with_grad(self):
        self.gradact = Arcsinh(2, 27.5, learn_lmbda=True)
        self.gradient = GradOp(k=2.0, learnable=False)
        self.compute_grad = True
        
    def init_tokenizer_state(self, img):
        B,C,H,W = img.shape
        V, E, mV, nV, byx = init_graph_topology(img)
        bb = torch.stack([byx[1],byx[2],byx[1],byx[2]], 0)
        fV, grad = self.preproc(img)
        s2 = torch.zeros_like(fV)
        info = torch.full_like(fV[:,0], -(H*W)**.5)
        curseg = V
        optnV = nV
        optinfo = info
        optseg = V
        optfV = fV
        opts2 = s2
        grid = GridDimensions(byx, B, C, H, W)
        state = TokenizerState(
            fV, s2, E, mV, info, curseg, optseg, optinfo, optfV, opts2,
            nV, optnV, grid, bb
        )
        return state, grad
    
    def mean_injection(self, img, res):
        fV = img.permute(0,2,3,1).reshape(-1, img.shape[1])
        replaced_mean = res.fV - scatter_mean_2d(fV, res.seg)
        return fV + replaced_mean[res.seg]
    
    def _refactor(self, res:TokenizerResult):
        B, H, W = res.seg.shape
        x = res.fV
        concat = 2*res.byx[1:].mT / x.new_tensor([H-1,W-1]) - 1
        if res.grad is not None:
            concat = torch.cat([
                concat, res.grad.permute(0,2,3,1).reshape(B*H*W, -1)
            ], -1)
        sizes = res.seg.view(-1).bincount()
        return torch.cat([x, concat], -1), None, res.seg, res.byx, sizes, res.nV, res.bb
            
    
    def forward(
        self, img, pretrain=False, final_merging=None, target=None, 
        alttok=False, max_it_override=None, residuals=False,
    ):
        if final_merging is None:
            final_merging = self.training
        state, grad = self.init_tokenizer_state(img)
        origE = state.E
        it = 0
        max_it = (
            int(np.ceil(np.log2((state.grid.H * state.grid.W)**.5)))
            if max_it_override is None else max_it_override
        )
        stop = False
        while (state.nV / state.grid.B > self.min_avg_region) and it < max_it and not stop:
            state, stop = update_tokenizer_state(
                state, it, self.similarity, self.criterion, 
                self.cmp, self.center, self.iota, self.eps,
                self.varproj
            )
            it += 1
        
        res = finalize_tokenization(
            state, origE, self.criterion, self.iota, self.eps,
            self.linear # self.proj
        )

        if final_merging:
            res = token_drop_merge(res, similarity=self.similarity, target=target)

        if pretrain:
            return res

        B, H, W = res.grid.B, res.grid.H, res.grid.W

        if residuals:
            residuals = img.permute(0,2,3,1) - scatter_mean_2d(
                img.permute(0,2,3,1).reshape(-1,img.shape[1]), 
                res.seg.view(-1)
            )[res.seg.view(B,H,W)]
            return residuals

        if alttok:
            x = self.mean_injection(img, res)
            concat = 2*res.grid.byx[1:].mT / x.new_tensor([H,W]) - 1
            if grad is not None:
                concat = torch.cat([
                    concat, grad.permute(0,2,3,1).reshape(B*H*W, -1)
                ], -1)
            sizes = res.seg.bincount()
            edges = get_seg_edges(res.seg, B, H, W)
            return torch.cat([x, concat], -1), res.seg.view(B,H,W), res.grid.byx, sizes, res.nV, res.bb

        return TokenizerResult(
            self.mean_injection(img, res),
            res.seg.view(B,H,W), res.grid.byx, res.bb, res.nV, grad
        )
    
    def compute_hierograph(self, img, max_it_override=None, override_stop_criterion=True, target=None):
        state, grad = self.init_tokenizer_state(img)
        origE = state.E
        it = 0
        max_it = (
            int(np.ceil(np.log2((state.grid.H * state.grid.W)**.5)))
            if max_it_override is None else max_it_override
        )
        stop = False
        segs = [state.curseg]
        while (state.nV / state.grid.B > self.min_avg_region) and it < max_it and not stop:
            state, stop = update_tokenizer_state(
                state, it, self.similarity, self.criterion, 
                self.cmp, self.center, self.iota, self.eps,
                self.varproj
            )
            if override_stop_criterion:
                stop = False
            segs.append(state.curseg)
            it += 1
        
        res = finalize_tokenization(
            state, origE, self.criterion, self.iota, self.eps,
            self.linear # self.proj
        )
        res = token_drop_merge(res, similarity=self.similarity, target=target)
        optseg = res.seg
        segs.append(res.grid.byx[0].view(res.seg.shape))

        return segs, optseg

