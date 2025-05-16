import torch
import torch.nn.functional as F

from torch import Tensor

in1k_mean = torch.tensor([0.485, 0.456, 0.406])
in1k_std = torch.tensor([0.229, 0.224, 0.225])

def in1k_norm(tensor:Tensor, dim:int=-1):
    '''Normalize a tensor using the ImageNet (in1k) mean and standard deviation.

    Parameters
    ----------
    tensor : Tensor
        The input tensor to be normalized.
    dim : int, optional
        The dimension along which the mean and std are applied. Default is -1.

    Returns
    -------
    Tensor
        The normalized tensor.

    Notes
    -----
    This function adjusts input data (tensor) to have zero mean and unit
    variance according to the ImageNet dataset statistics.
    '''

    shape = [1] * tensor.ndim
    shape[dim] = -1
    mean = in1k_mean.view(shape).to(tensor.device)
    std = in1k_std.reshape(shape).to(tensor.device)
    return (tensor - mean) / std


def in1k_unnorm(tensor:Tensor, dim=-1):
    '''Un-normalize a tensor using the ImageNet (in1k) mean and standard deviation.

    Parameters
    ----------
    tensor : Tensor
        The input tensor to be un-normalized.
    dim : int, optional
        The dimension along which the mean and std are applied. Default is -1.

    Returns
    -------
    Tensor
        The un-normalized tensor.

    Notes
    -----
    This function reverses the normalization effect by applying the original
    ImageNet mean and standard deviation to the normalized tensor.
    '''

    shape = [1] * tensor.ndim
    shape[dim] = -1
    mean = in1k_mean.view(shape).to(tensor.device)
    std = in1k_std.reshape(shape).to(tensor.device)
    return tensor * std + mean


def image_gradient(img:torch.Tensor, unnorm:bool=True) -> torch.Tensor:
    '''Computes gradient (Scharr) features of an image.

    Parameters
    ----------
    img : Tensor
        Image tensor.
    unnorm : bool, optional
        Flag for unnormalization. Defaults to True.


    Returns
    -------
    Tensor
        Gradient features.
    '''
    if unnorm:
        img = in1k_unnorm(img, 1)
    img = (0.299 * img[:,0] + 0.587 * img[:,1] + 0.114 * img[:,2]).unsqueeze(1)
    kernel = img.new_tensor([[[[-3.,-10,-3.],[0.,0.,0.],[3.,10,3.]]]])
    kernel = torch.cat([kernel, kernel.mT], dim=0)
    return F.conv2d(
        F.pad(img, 4*[1], mode='replicate'), 
        kernel, 
        stride=1
    ).div_(16)