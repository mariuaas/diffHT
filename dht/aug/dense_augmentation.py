import torchvision as tv
import torchvision.transforms as T
import torchvision.transforms.v2 as v2
import math
import numpy as np
import quix

from typing import Optional, Callable

class Unsqueeze0:

    def __call__(self, x):
        return x[None]

class RandAugment_v2:

    def __init__(self, img_size, num_transforms, magnitude, num_bins=31):

        affine_kwargs = dict(
            degrees=(0.0,0.0),
            translate=[0,0],
            scale=(1.0,1.0),
            shear=[0, 0],
            interpolation=T.InterpolationMode.BILINEAR,
        )
        self.num_transforms = num_transforms
        shear_kwargs = {k:v for k,v in affine_kwargs.items() if k != 'shear'}
        translate_kwargs = {k:v for k,v in affine_kwargs.items() if k != 'translate'}
        rotate_kwargs = {k:v for k,v in affine_kwargs.items() if k != 'degrees'}

        self.shear_x_mag = v2.RandAugment._AUGMENTATION_SPACE['ShearX'][0](num_bins, img_size, img_size)[magnitude] # Signed
        self.shear_y_mag = v2.RandAugment._AUGMENTATION_SPACE['ShearY'][0](num_bins, img_size, img_size)[magnitude] # Signed
        self.translate_x_mag = v2.RandAugment._AUGMENTATION_SPACE['TranslateX'][0](num_bins, img_size, img_size)[magnitude] # Signed
        self.translate_y_mag = v2.RandAugment._AUGMENTATION_SPACE['TranslateY'][0](num_bins, img_size, img_size)[magnitude] # Signed
        self.rotate_mag = v2.RandAugment._AUGMENTATION_SPACE['Rotate'][0](num_bins, img_size, img_size)[magnitude] # Signed
        self.brightness_mag = v2.RandAugment._AUGMENTATION_SPACE['Brightness'][0](num_bins, img_size, img_size)[magnitude] # Signed
        self.color_mag = v2.RandAugment._AUGMENTATION_SPACE['Color'][0](num_bins, img_size, img_size)[magnitude] # Signed
        self.contrast_mag = v2.RandAugment._AUGMENTATION_SPACE['Contrast'][0](num_bins, img_size, img_size)[magnitude] # Signed
        self.sharpness_mag = v2.RandAugment._AUGMENTATION_SPACE['Sharpness'][0](num_bins, img_size, img_size)[magnitude] # Signed
        self.posterize_mag = v2.RandAugment._AUGMENTATION_SPACE['Posterize'][0](num_bins, img_size, img_size)[magnitude] # Unsigned
        self.solarize_mag = v2.RandAugment._AUGMENTATION_SPACE['Solarize'][0](num_bins, img_size, img_size)[magnitude] # Unsigned

        self.shear_x_mag = math.degrees(math.atan(self.shear_x_mag))
        self.shear_y_mag = math.degrees(math.atan(self.shear_y_mag))
        self.translate_x_mag = self.translate_x_mag / img_size
        self.translate_y_mag = self.translate_y_mag / img_size
        
        self.transform_dict = dict(
            identity = self.identity,
            shear_x = v2.RandomAffine(shear=[-self.shear_x_mag, self.shear_x_mag, 0, 0], center=[0,0], **shear_kwargs), # type: ignore
            shear_y = v2.RandomAffine(shear=[0, 0, -self.shear_y_mag, self.shear_y_mag], center=[0,0], **shear_kwargs), # type: ignore
            translate_x = v2.RandomAffine(translate=[self.translate_x_mag, 0], **translate_kwargs), # type: ignore
            translate_y = v2.RandomAffine(translate=[0, self.translate_y_mag], **translate_kwargs), # type: ignore
            rotate = v2.RandomAffine(degrees=(-self.rotate_mag, self.rotate_mag), **rotate_kwargs), # type: ignore
            brightness = v2.ColorJitter(brightness=(1.0-self.brightness_mag, 1.0+self.brightness_mag)),
            color = v2.ColorJitter(saturation=(1.0-self.color_mag, 1.0+self.color_mag)), 
            contrast = v2.ColorJitter(contrast=(1.0-self.contrast_mag, 1.0+self.contrast_mag)), 
            sharpness = v2.RandomChoice([
                v2.RandomAdjustSharpness(sharpness_factor=1.0 - self.sharpness_mag),
                v2.RandomAdjustSharpness(sharpness_factor=1.0 + self.sharpness_mag),
            ]),
            posterize = v2.RandomPosterize(bits=self.posterize_mag, p=1.0),
            solarize = v2.RandomSolarize(threshold=self.solarize_mag, p=1.0),
            autocontrast = v2.RandomAutocontrast(p=1.0),
            equalize = v2.RandomEqualize(p=1.0),
            invert = v2.RandomInvert(p=1.0),
        )

    @staticmethod
    def identity(args):
        return args

    def __call__(self, *args):
        transform_keys = np.random.choice(list(self.transform_dict.keys()), size=self.num_transforms, replace=False)
        return v2.Compose([
            self.transform_dict[key] for key in transform_keys
        ])(*args)


class MainAugment:
    
    RANDAUG_DICT = {
        'none': (0, 0),
        'light': (2, 10),
        'medium': (2, 15),
        'strong': (2, 20),
    }

    def __init__(self, img_size:int, randaug_strength:str='medium', use_aug3:bool=True, num_bins:int=31):
        aug3 = v2.RandomChoice([
            v2.RandomSolarize(0.5, 1.0),
            v2.ColorJitter(0, 0, (0., 1.5), 0), # Saturation instead of grayscale
            v2.GaussianBlur(int(img_size * .1) | 1)
        ])

        if randaug_strength not in self.RANDAUG_DICT:
            raise ValueError(f'Got unknown RandAug strength: `{randaug_strength}`!')

        if randaug_strength != 'none':
            num_transforms, magnitude = self.RANDAUG_DICT[randaug_strength]
            randaug = RandAugment_v2(img_size, num_transforms, magnitude, num_bins=num_bins)
            if use_aug3:
                self.augment = v2.RandomChoice([aug3, randaug])
            else:
                self.augment = randaug
        elif use_aug3:
            self.augment = aug3
        else:
            self.augment = self.identity

    @staticmethod
    def identity(args):
        return args

    def __call__(self, args):
        return self.augment(args)
    


def parse_dense_augmentation(cfg:quix.DataConfig, num_classes:Optional[int]=None) -> tuple[Callable,Callable]:
    intp_modes = [
        quix.data.aug.INTERPOLATION_MODES[m] for m in cfg.intp_modes 
        if m in quix.data.aug.INTERPOLATION_MODES
    ]
    identity = quix.data.aug.Identity()

    # Randaug + Aug3
    mainaug = MainAugment(cfg.img_size, cfg.randaug, cfg.aug3)

    # Random resize crop (randomize interpolations)
    resizecrop = v2.RandomChoice([
        v2.RandomResizedCrop(
            cfg.img_size, cfg.rrc_scale, cfg.rrc_ratio, 
            interpolation=itm, antialias=True
        )
        for itm in intp_modes
    ])

    # Drop Cutmix + Mixup for dense traning
    batch_augs = identity

    # Final augmentations
    addaug = [
        v2.ColorJitter(0.3, 0.3) if cfg.jitter else identity,                               # Only brightness and contrast
        v2.RandomHorizontalFlip() if cfg.hflip else identity,                               # On by default
        v2.RandomVerticalFlip() if cfg.vflip else identity,                                 # Off by default
        v2.Normalize(mean=cfg.rgb_mean, std=cfg.rgb_std) if cfg.use_rgb_norm else identity  # On by default,
    ]

    sample_augs = v2.Compose([resizecrop, mainaug, *addaug])
    return sample_augs, quix.data.aug.DefaultCollateWrapper(batch_augs)
