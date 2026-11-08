"""Preprocessing used in the thesis: Reinhard LAB colour harmonisation + ImageNet normalisation.

The harmonisation reference statistics were computed from 400 training images (seed 42) of the merged
26-class dataset. Use the same values at inference time so new images look like the training data.
"""
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Reference LAB statistics from the thesis run (training split, seed 42), OpenCV 8-bit LAB scale.
REF_LAB_MEAN = np.array([136.71373, 118.08872, 148.07559], dtype=np.float32)
REF_LAB_STD = np.array([53.228355, 11.685024, 13.002322], dtype=np.float32)


class ReinhardHarmonize:
    """Deterministic transform that shifts an image's LAB colour statistics toward the shared reference."""

    def __init__(self, ref_mean=REF_LAB_MEAN, ref_std=REF_LAB_STD):
        self.ref_mean = ref_mean
        self.ref_std = ref_std

    def __call__(self, pil_img):
        import cv2
        arr = np.array(pil_img.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        src_mean, src_std = lab.mean(axis=(0, 1)), lab.std(axis=(0, 1)) + 1e-6
        lab = (lab - src_mean) / src_std * self.ref_std + self.ref_mean
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def get_train_transform(img_size=224, harmonize=None):
    harmonize = harmonize or ReinhardHarmonize()
    return transforms.Compose([
        harmonize,
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 4 / 3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(img_size=224, harmonize=None):
    harmonize = harmonize or ReinhardHarmonize()
    return transforms.Compose([
        harmonize,
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
