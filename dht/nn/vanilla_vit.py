import torch
import torch.nn as nn

from .transformer import DPXEncoder
from ..nn.cnn import GradOp, Arcsinh
from ..tok.tokenizer import TokenizerResult
from ..tok.extractor import (
    DPXExtractorResult, GradientHistogramExtractor, 
    PositionalHistogramExtractor, InterpolationExtractor
)
from ..tok.embedder import DPXResult
from ..utils.indexing import unravel_index
from ..utils.segmentation import uniform_square_partitions
from ..utils.clstools import update_signature_kwargs

class ViTPatchTokenizer(nn.Module):

    def __init__(
        self, embed_dim, patch_size, channels=3, compute_grad=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.compute_grad = compute_grad
        if compute_grad:
            self.gradact = Arcsinh(2, 0, learn_lmbda=True)
            self.gradient = GradOp(k=2.0, learnable=False)
        else:
            self.gradact = nn.Identity()
            self.gradient = nn.Identity()


    def get_bbox(self, seg, byx, nV):
        _, y, x = byx
        bbox_out = seg.new_zeros(4, nV)
        bbox_out[0].scatter_reduce_(0, seg.view(-1), y, 'amin', include_self=False)
        bbox_out[1].scatter_reduce_(0, seg.view(-1), x, 'amin', include_self=False)
        bbox_out[2].scatter_reduce_(0, seg.view(-1), y, 'amax', include_self=False)
        bbox_out[3].scatter_reduce_(0, seg.view(-1), x, 'amax', include_self=False)
        return bbox_out

    def forward(self, img, **kwargs):
        B,C,H,W = img.shape
        grad = None
        if self.compute_grad:
            grad = self.gradact(self.gradient(img))
        seg = uniform_square_partitions(B,H,W,self.patch_size,img.device)
        nV = int(seg.max().item()) + 1
        byx = unravel_index(torch.arange(B*H*W, device=img.device), (B,H,W))
        fV = img.permute(0,2,3,1).reshape(-1,self.channels)
        bb = self.get_bbox(seg, byx, nV)
        return TokenizerResult(
            fV, seg, byx, bb, nV, grad
        )


class ViTPatchExtractor(nn.Module):

    def __init__(
        self, patch_size, channels=3, mix_init=0.01
    ):
        super().__init__()
        self.patch_size = patch_size
        self.extractor_feat = InterpolationExtractor(patch_size, channels, False)
        self.extractor_pos = PositionalHistogramExtractor(patch_size)
        self.extractor_grad = GradientHistogramExtractor(patch_size)
        mix = ((1/mix_init-1)*torch.ones(1)).log().neg()
        self.register_buffer("_mix", mix, persistent=True) # Unused
        _pmt = nn.Parameter(1e-1 * torch.randn(channels, patch_size, patch_size))
        self.register_buffer('pixel_mask_token', _pmt, persistent=True) # Unused
        self.use_kde = False

    @property
    def mix(self):
        return self._mix.sigmoid()
    
    @mix.setter
    def mix(self, val):
        self._mix.data = ((1/val-1)*torch.ones(1)).log().neg()

    def forward(self, res:TokenizerResult):
        feat = self.extractor_feat(res)
        pos = self.extractor_pos(res)
        grd = self.extractor_grad(res)
        posgrad = torch.cat([pos] + ([] if grd is None else [grd]), 1)
        return DPXExtractorResult(
            torch.cat([feat, posgrad], 1), res.seg, res.byx, res.nV
        )


class ViTPatchEmbedder(nn.Module):

    def __init__(
        self, embed_dim, patch_size, 
        channels=3, compute_grad=False, num_cls_tokens=1, embed_bias=True, pre_norm=True            
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_cls_tokens = num_cls_tokens
        self.channels = channels
        self.compute_grad = compute_grad
        self.token_dim = (channels + 1 + compute_grad) * patch_size**2
        self.proj = nn.Linear(self.token_dim, embed_dim, bias=embed_bias)
        self.cls_token = nn.Parameter(
            1e-4 * torch.randn(self.num_cls_tokens, embed_dim)
        )
        self.norm_pre = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()

    def forward(self, res:DPXExtractorResult):
        fV, seg, byx, nV = res
        B,H,W = seg.shape
        T, nC, D = self.token_dim, self.num_cls_tokens, self.embed_dim
        assert nV % B == 0, f"Error in num total patches nV % B == {nV % B}"
        fV = torch.cat([self.cls_token.view(1,nC,D).expand(B,nC,D), self.proj(fV.view(B,-1,T))], 1)
        amask = seg.new_ones(*fV.shape[:2], dtype=torch.bool)
        return DPXResult(self.norm_pre(fV), seg, amask)
    

class ViTEncoder(DPXEncoder):

    def __init__(self, embed_dim, patch_size, heads, depth, **kwargs):
        super().__init__(embed_dim, patch_size, heads, depth, **kwargs)
        self.tokenizer = ViTPatchTokenizer(
            embed_dim, patch_size,
            **update_signature_kwargs(
                ViTPatchTokenizer, **kwargs
            )            
        )
        self.extractor = ViTPatchExtractor(
            patch_size,
            **update_signature_kwargs(
                ViTPatchExtractor, **kwargs
            )
        )
        self.embedder = ViTPatchEmbedder(
            embed_dim, patch_size, 
            **update_signature_kwargs(
                ViTPatchEmbedder, **kwargs
            )
        )

class ViTClassifier(ViTEncoder):

    def __init__(self, embed_dim, patch_size, heads, depth, n_classes=1000, **kwargs):
        super().__init__(embed_dim, patch_size, heads, depth, **kwargs)
        self.n_classes = n_classes
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, final_merging=None, target=None, encode_only=False):
        x, seg, amask = super().forward(x, final_merging, target)
        if encode_only:
            return x[:,0], seg, amask
        return self.head(x[:,0])
