import torch
import torch.nn.functional as F
import cupy
import numpy as np
import os
from numba import njit, prange

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

with open(os.path.join(__location__, 'src', 'scatterhist.cu'), 'r') as sch_file:
    _scatterhist_kernel_code = sch_file.read()

_scatter_joint_hist_kernel = cupy.RawKernel(
    _scatterhist_kernel_code, 'scatter_joint_hist'
)


@njit(parallel=True)
def _scatter_joint_hist_cpu(seg, feats, mesh_y, mesh_x, featcombs, output, sigma, n, nbins, nfeats, feat_dim):
    nbins2 = nbins * nbins
    for idx in prange(n):
        s_idx = seg[idx]
        for j in range(nfeats):
            j_y = featcombs[j,0]
            j_x = featcombs[j,1]
            y = feats[idx,j_y]
            x = feats[idx,j_x]
            
            for i in range(nbins2):
                z1 = (y - mesh_y[i]) / sigma
                z2 = (x - mesh_x[i]) / sigma
                value = np.exp(-0.5 * (z1 * z1 + z2 * z2))
                output[s_idx, j, i] += value


def scatter_joint_hist(
    seg:torch.Tensor, feats:torch.Tensor, num_seg, num_bins, featcombs,
    sigma=0.025, low=-1, high=1,
    tpb=1024,
):
    device = feats.device
    n, feat_dim = feats.shape
    delta = 1/num_bins
    featcombs = seg.new_tensor(featcombs)
    num_feats = len(featcombs)

    assert n == len(seg), f"{n} != {len(seg)}"
    assert featcombs.max() < feat_dim, f'{featcombs.max().item()=}>={feat_dim=}'
    assert featcombs.min() >= 0
    assert feats.dtype == torch.float
        
    bins1d = torch.linspace(low+delta, high-delta, num_bins, device=seg.device)
    mesh_y, mesh_x = [mesh.flatten() for mesh in torch.meshgrid(bins1d, bins1d, indexing='ij')] # type:ignore
    output = bins1d.new_zeros(num_seg, num_feats, num_bins**2)
    sigmaptr = bins1d.new_tensor([sigma])
    
    _tonp = lambda x: x.numpy()
    _todl = cupy.from_dlpack
    if device.type == 'cpu':
        _scatter_joint_hist_cpu(
            _tonp(seg), 
            _tonp(feats.detach().contiguous()),
            _tonp(mesh_y),
            _tonp(mesh_x),
            _tonp(featcombs),
            _tonp(output),
            sigma, n, num_bins, num_feats, feat_dim
        )
    else:
        bpg = (n + tpb - 1) // tpb
        with cupy.cuda.Device(seg.device.index) as cpdev:
            _scatter_joint_hist_kernel(
                (bpg,), (tpb,), (
                    _todl(seg), 
                    _todl(feats.detach().contiguous()),
                    _todl(mesh_y),
                    _todl(mesh_x),
                    _todl(featcombs),
                    _todl(output),
                    _todl(sigmaptr),
                    n, num_bins, num_feats, feat_dim
                )
            )
    
    return output.view(num_seg, -1)
    