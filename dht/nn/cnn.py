import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):

    def __init__(
        self, in_ch, out_ch, ksize, pad, stride, 
        act:nn.Module=nn.SiLU(inplace=True), bias=True, norm=False
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ksize, stride, pad, bias=bias),
            nn.BatchNorm2d(out_ch) if norm else nn.Identity(),
            act,
        )

    def forward(self, x):
        return self.conv(x)


class DenseLayer(nn.Module):

    def __init__(self, in_ch, out_ch, act:nn.Module=nn.Identity(), norm=False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch) if norm else nn.Identity(),
            act,
        )

    def forward(self, x):
        return self.fc(x)


class Downsample(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        _id = nn.Identity()
        self.conv = nn.Sequential(
            ConvLayer(in_ch, in_ch, 2, 0, 2, bias=False),
            ConvLayer(in_ch, out_ch, 3, 1, stride, bias=False),
        )

    def forward(self, x):
        return self.conv(x)


class Pointwise(nn.Module):

    def __init__(self, hid_ch=16):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer(3, hid_ch, 1, 0, 1, norm=True),
            ConvLayer(hid_ch, 1, 1, 0, 1, act=nn.Tanh()),
        )
        
    def forward(self, x):
        return self.conv(x)


class Slice(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grid, guide):
        B, C, H, W = guide.shape
        dev = guide.device
        dtp = guide.dtype
        Hmul, Wmul = 2/(H-1), 2/(W-1)
        y, x = torch.meshgrid([
            torch.linspace(0, H-1, H, device=dev, dtype=dtp).mul_(Hmul).sub_(1), 
            torch.linspace(0, W-1, W, device=dev, dtype=dtp).mul_(Wmul).sub_(1),
        ], indexing='ij')
        y = y.view(1, *y.shape, 1).expand(B, *y.shape, 1)
        x = x.view(1, *x.shape, 1).expand(B, *x.shape, 1)
        # NOTE: This order seem weird, but the poorly documented F.grid_sample
        #       actually requires x, y, z order for these dimensions.
        guide = torch.cat([x, y, guide.permute(0,2,3,1)], dim=-1).unsqueeze(1) 
        coeff = F.grid_sample(grid, guide, align_corners=True)
        return coeff.squeeze(2)


class Affine(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.unflat = nn.Unflatten(1, (in_ch+1, out_ch))

    def forward(self, coeff, full_res_input):
        return torch.einsum(
            'bchw,bcdhw->bdhw',
            F.pad(full_res_input, (0,0,0,0,0,1), value=1),
            self.unflat(coeff)
        )

class BilateralNet(nn.Module):

    def __init__(self, in_ch, out_ch, ds_size=(256,256), quantization=16, act=nn.SiLU(inplace=True)):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        _id = nn.Identity()
        self.act = act
        D = out_ch*(in_ch + 1)
        Q = quantization
        self.splat = nn.Sequential(
            nn.Upsample(ds_size, mode='bilinear', align_corners=True),
            Downsample(in_ch,8,1),
            Downsample(8,16,1),
            Downsample(16,32,1),
            Downsample(32,64,1),
        )
        self.glob = nn.Sequential(
            Downsample(64, 64, 1),
            Downsample(64, 64, 1),
            nn.Flatten(-3,-1),
            DenseLayer(1024, 256, act, True),
            DenseLayer(256, 128, act, True),
            DenseLayer(128, 64, _id, False),
            nn.Unflatten(-1, (64,1,1)),
        )
        self.loc = nn.Sequential(
            ConvLayer(64, 64, 3, 1, 1, act, True, True),
            ConvLayer(64, 64, 3, 1, 1, _id, False, False),
        )
        self.proj = nn.Sequential(
            act,
            nn.Conv2d(64, D*Q, 1),
            nn.Unflatten(1, (D, Q))
        )
        self.guidefn = Pointwise()
        self.slicefn = Slice()
        self.affinefn = Affine(in_ch, out_ch)
        
    def forward(self, x):
        spl = self.splat(x)
        coeff = self.slicefn(
            self.proj(self.glob(spl) + self.loc(spl)), 
            self.guidefn(x)
        )
        return self.affinefn(coeff, x)



class Arcsinh(nn.Module):
    def __init__(self, in_ch, lmbda_init, learn_lmbda=True):
        super().__init__()
        if learn_lmbda:
            self.lmbda = nn.Parameter((lmbda_init*torch.ones(in_ch)).view(-1,1,1))
        else:
            self.register_buffer('lmbda', (lmbda_init*torch.ones(in_ch)).view(-1,1,1))

    def forward(self, x):
        m, d = self.lmbda, torch.arcsinh(self.lmbda)
        pos = self.lmbda > 0
        neg = self.lmbda < 0

        pos_output = x.mul(m).arcsinh().div(d)
        neg_output = x.mul(d).sinh().div(m)
        zro_output = x

        output = torch.where(pos, pos_output, torch.where(neg, neg_output, zro_output))
        return output
    

class HighBoost(nn.Module):

    def __init__(self, k=1.0, learnable=False):
        super().__init__()
        kernel = torch.ones(1,1,3,3)
        kernel[...,1,1] = 1 - kernel.sum()
        if learnable:
            self.k = nn.Parameter(k*torch.ones(1))
            self.kernel = nn.Parameter(kernel)
        else:
            self.register_buffer('k', k*torch.ones(1))
            self.register_buffer('kernel', kernel)

    def normalized_kernel(self):
        norm = self.kernel.abs().sum(dim=-1).sum(dim=-1)
        return self.kernel / norm[...,None,None]

    def forward(self, x):
        B, C, H, W = x.shape
        lap = F.conv2d(
            F.pad(x, (1,1,1,1), mode='replicate'), 
            self.normalized_kernel().expand(C,1,3,3), 
            groups=C
        )
        return x - self.k*lap


class GradOp(nn.Module):
    
    def __init__(self, learnable=False, **kwargs):
        super().__init__()
        kernel = torch.tensor([[[
            [-3., -10., -3.], 
            [ 0.,   0.,  0.], 
            [ 3.,  10.,  3.]
        ]]])
        if learnable:
            self.kernel = nn.Parameter(kernel)
        else:
            self.register_buffer('kernel', kernel)

    def normalized_kernel(self):
        norm = self.kernel.abs().sum(dim=-1).sum(dim=-1)
        return self.kernel / norm[...,None,None]

    def forward(self, x):
        kernel = self.normalized_kernel()
        return F.conv2d(
            F.pad(x.mean(dim=1, keepdim=True), 4*[1], mode='replicate'), 
            torch.cat([kernel, kernel.mT], 0)
        )
    

def SimpleConv(hidden):
    return nn.Sequential(
        nn.Conv2d(5, hidden, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, hidden, 1, 1, 0, 1, hidden)
    )


class SimpleDownsample(nn.Module):
    def __init__(self, in_ch, hid_ch, ratio=4.0):
        super().__init__()
        r = int(round(ratio))
        self.ds = nn.Sequential(
            nn.Conv2d(in_ch, hid_ch, 1, 1, 0),
            nn.BatchNorm2d(hid_ch),
        )
        self.cv = nn.Sequential(
            nn.Conv2d(in_ch, r*hid_ch, 2, 2),
            nn.BatchNorm2d(r*hid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(r*hid_ch, hid_ch, 2, 2),
            nn.BatchNorm2d(hid_ch),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.act = nn.Sequential(
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.act(
            self.ds(x) + self.cv(x)
        )