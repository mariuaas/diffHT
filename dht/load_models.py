import torch
import torch.nn as nn
from .nn.transformer import DPXClassifier, DPXDenseModel
from .nn.vanilla_vit import ViTClassifier
from typing import Optional

def load_model_path(
    tokenizer:str,
    capacity:str,
    patch_size:int,        
    transfer:bool
) -> str:
    rootpath = '/work2/mariuaas/'
    modeljoin = capacity.upper() + str(patch_size)
    path = None
    if tokenizer == 'dht':
        if transfer:
            path = dict(
                S16 = 'dpx/dpx_vit_S16_classifier_simpledown_nograd_nokde_qknorm_c0_i3_h8.pth',
                B16 = 'dpx/dpx_vit_B16_classifier_simpledown_nograd_nokde_qknorm_c0_i3_h8.pth',
                B32 = 'dpx/dpx_vit_B32_classifier_simpledown_nograd_nokde_qknorm_c0_i3_h8.pth',
            ).get(modeljoin, None)
        else:
            path = dict(
                S16 = 'ASTra/final/s16_full_seg_ft_pos24.pth',
                B16 = 'ASTra/final/b16_full_cls_ft_pos24.pth',
            ).get(modeljoin, None)
    elif tokenizer == 'vanilla':
        raise NotImplementedError()
    if path is None:    
        raise ValueError('Invalid arguments to load_model_path!')
    return rootpath + path


def load_classification_model(
    tokenizer:str,
    capacity:str,
    patch_size:int,
    transfer:bool = False,
    pretrained:bool = True,
    interpolate_to:Optional[int] = None,
    qk_norm=True, 
    kde=False, 
    cmp=0.1, 
    iota=5, 
    compute_grad=True, 
    normalize_interpolation=False,
    tome_ratios=[0], 
    dop_path=0., 
    pos_patch_size=24, 
    pos_patch_scale=16,
    cnn_backbone='downsample', 
    tokenizer_hidden=8, 
) -> nn.Module:
    if transfer:
        compute_grad = False
    kwargs = dict(
        qk_norm = qk_norm, 
        kde = kde, 
        cmp = cmp, 
        iota = iota,
        compute_grad = compute_grad, 
        normalize_interpolation = normalize_interpolation,
        tome_ratios = tome_ratios, 
        dop_path = dop_path,
        pos_patch_size = pos_patch_size, 
        pos_patch_scale = pos_patch_scale,
        cnn_backbone=cnn_backbone,
        tokenizer_hidden=tokenizer_hidden,
    )
    assert tokenizer in ['vanilla', 'dht']
    assert capacity.lower() in ['s', 'b']
    assert patch_size in [16, 32]
    if patch_size == 32:
        assert transfer or (tokenizer == 'vanilla')

    if tokenizer == 'dht':
        model = DPXClassifier.build(
            capacity, patch_size, **kwargs
        )
        if pretrained:
            path = load_model_path(
                tokenizer, capacity, patch_size, transfer
            )
            sd = torch.load(path, map_location='cpu')
            sd = sd.get('model', sd)
            model.load_state_dict(sd)
            model._dhtloadpath = path # type: ignore
        
        if interpolate_to is not None:
            model.interpolate_pos_embed(interpolate_to, 16)
        return model
    
    elif tokenizer == 'vanilla':
        if transfer:
            try:
                import timm
            except ImportError:
                raise ValueError('timm required for vanilla transfer models.')
            modeljoin = capacity.upper() + str(patch_size)
            timm_model = dict(
                B16 = 'deit3_base_patch16_224',
                S16 = 'deit3_small_patch16_224',
                B32 = 'vit_base_patch32_224'
            ).get(modeljoin, None)

            if timm_model is None:
                raise ValueError(f'Invalid timm model {modeljoin}')
            
            model = timm.create_model(timm_model, pretrained=pretrained)
            return model
        else:
            if interpolate_to is not None:
                raise NotImplementedError('Hold yer horses, cowboy...')
            model = ViTClassifier.build(
                capacity, patch_size, **kwargs
            )
            if pretrained:
                path = load_model_path(
                    tokenizer, capacity, patch_size, transfer
                )
                sd = torch.load(path, map_location='cpu')
                sd = sd.get('model', sd)
                model.load_state_dict(sd)
                model._dhtloadpath = path # type: ignore

            return model
        
    raise ValueError('Cannot load model config.')



