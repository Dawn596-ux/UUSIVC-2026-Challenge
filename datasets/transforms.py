# datasets/transforms.py

from typing import Tuple, Optional
import numpy as np
import torch
from PIL import Image


def ensure_3ch(image: np.ndarray) -> np.ndarray:
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
        [H, W, 3]
    """
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.concatenate([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        # Some PNG inputs are RGBA; drop the alpha channel for RGB-only models.
        image = image[..., :3]
    elif image.ndim == 3 and image.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return image


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    image: [H, W, C]
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    return np.array(pil_image.resize(size, resample=Image.BILINEAR))


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    mask: [H, W]
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    pil_mask = Image.fromarray(mask)
    return np.array(pil_mask.resize(size, resample=Image.NEAREST))


def normalize_to_float(image: np.ndarray) -> np.ndarray:
    """
    The organizers keep this baseline note in English for public release.
    """
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def mask_255_to_01(mask: np.ndarray) -> np.ndarray:
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    """
    mask = (mask > 0).astype(np.uint8)
    return mask


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    [H, W, C] -> [C, H, W]
    """
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """
    [H, W] -> [H, W]
    """
    return torch.from_numpy(mask).long()


def _add_speckle_noise(img: np.ndarray, **kwargs) -> np.ndarray:
    """乘性高斯噪声，模拟超声斑点噪声（speckle）。

    超声图像的噪声本质是乘性的（信号越强噪声越大），用均值 1、标准差 0.05
    的高斯噪声逐像素相乘近似。
    """
    noise = np.random.normal(1.0, 0.05, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) * noise, 0, 255).astype(np.uint8)


def build_augmentation():
    """构建训练用 albumentations 增强管线（几何 + 强度 + 超声斑点噪声）。

    惰性 import albumentations（本地不装该依赖，仅服务器训练环境需要）。
    几何增强通过 albumentations 的 mask 同步机制保证 mask 与 image 一致变换。
    """
    import albumentations as A

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15,
            border_mode=0, value=0, mask_value=0, p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.Lambda(image=_add_speckle_noise, p=0.5),
    ])


class BasicImageTransform:
    def __init__(self, output_size=(224, 224), binary_mask: bool = True, augmentation=None):
        self.output_size = output_size
        self.binary_mask = binary_mask
        self.augmentation = augmentation

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        image = ensure_3ch(image)
        image = resize_image(image, self.output_size)

        if mask is not None:
            if self.binary_mask:
                mask = mask_255_to_01(mask)
            else:
                mask = mask.astype(np.uint8)
            mask = resize_mask(mask, self.output_size)

        if self.augmentation is not None:
            if mask is not None:
                aug = self.augmentation(image=image, mask=mask)
                image, mask = aug["image"], aug["mask"]
            else:
                aug = self.augmentation(image=image)
                image = aug["image"]

        image = normalize_to_float(image)

        image_tensor = image_to_tensor(image)

        if mask is None:
            return image_tensor, None

        mask_tensor = mask_to_tensor(mask)
        return image_tensor, mask_tensor


class BasicVideoTransform:
    def __init__(self, output_size=(224, 224), binary_mask: bool = True, augmentation=None):
        self.output_size = output_size
        self.image_tf = BasicImageTransform(output_size, binary_mask=binary_mask, augmentation=augmentation)

    def __call__(self, frames, masks=None):
        """
        frames: list[np.ndarray], len=T
        masks: list[np.ndarray] or None
        returns:
            frame_tensor: [T, C, H, W]
            mask_tensor: [T, H, W] or None
        """
        frame_tensors = []
        mask_tensors = []

        if masks is None:
            for frame in frames:
                frame_tensor, _ = self.image_tf(frame, None)
                frame_tensors.append(frame_tensor)
            return torch.stack(frame_tensors, dim=0), None

        assert len(frames) == len(masks), "frames and masks must have same length"
        for frame, mask in zip(frames, masks):
            frame_tensor, mask_tensor = self.image_tf(frame, mask)
            frame_tensors.append(frame_tensor)
            mask_tensors.append(mask_tensor)

        return torch.stack(frame_tensors, dim=0), torch.stack(mask_tensors, dim=0)
