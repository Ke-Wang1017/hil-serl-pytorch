from functools import partial
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Tuple, Optional


def random_crop(img: torch.Tensor, padding: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Apply random crop to image with padding."""
    if generator is None:
        generator = torch.Generator()
        
    crop_from = torch.randint(0, 2 * padding + 1, (2,), generator=generator)
    crop_from = torch.cat([crop_from, torch.zeros(1, dtype=torch.int64)])
    
    padded_img = F.pad(img, (padding,) * 4, mode='replicate')
    return padded_img[
        crop_from[0]:crop_from[0] + img.shape[0],
        crop_from[1]:crop_from[1] + img.shape[1],
        :
    ]


def batched_random_crop(img: torch.Tensor, padding: int, num_batch_dims: int = 1, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Apply random crop to batched images."""
    # Flatten batch dims
    original_shape = img.shape
    img = img.reshape(-1, *img.shape[num_batch_dims:])
    
    # Process each image in batch
    crops = []
    for i in range(img.shape[0]):
        crops.append(random_crop(img[i], padding, generator))
    img = torch.stack(crops)
    
    # Restore batch dims
    return img.reshape(original_shape)


def resize(image: torch.Tensor, image_dim: Tuple[int, int]) -> torch.Tensor:
    """Resize image to given dimensions."""
    assert len(image_dim) == 2
    return F.interpolate(image.unsqueeze(0), size=image_dim, mode='bilinear', align_corners=False).squeeze(0)


def _maybe_apply(apply_fn, inputs: torch.Tensor, apply_prob: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Conditionally apply function with given probability."""
    if generator is None:
        generator = torch.Generator()
        
    if torch.rand(1, generator=generator).item() <= apply_prob:
        return apply_fn(inputs)
    return inputs


def rgb_to_hsv(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert RGB to HSV color space."""
    img = torch.stack([r, g, b], dim=-1)
    return TF.rgb_to_hsv(img).unbind(dim=-1)


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert HSV to RGB color space."""
    img = torch.stack([h, s, v], dim=-1)
    return TF.hsv_to_rgb(img).unbind(dim=-1)


def adjust_brightness(rgb_tuple: Tuple[torch.Tensor, ...], delta: float) -> Tuple[torch.Tensor, ...]:
    """Adjust brightness of RGB image."""
    return tuple(x + delta for x in rgb_tuple)


def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust contrast of image."""
    mean = torch.mean(image, dim=(-2, -1), keepdim=True)
    return factor * (image - mean) + mean


def adjust_saturation(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, factor: float) -> Tuple[torch.Tensor, ...]:
    """Adjust saturation in HSV color space."""
    return h, torch.clamp(s * factor, 0.0, 1.0), v


def adjust_hue(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, delta: float) -> Tuple[torch.Tensor, ...]:
    """Adjust hue in HSV color space."""
    return (h + delta) % 1.0, s, v


def color_transform(
    image: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    to_grayscale_prob: float = 0.0,
    color_jitter_prob: float = 1.0,
    apply_prob: float = 1.0,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply color jittering to image."""
    if generator is None:
        generator = torch.Generator()
    
    if torch.rand(1, generator=generator).item() > apply_prob:
        return image
        
    transforms = []
    if brightness > 0:
        transforms.append(lambda img: TF.adjust_brightness(img, 1 + torch.rand(1, generator=generator).item() * brightness))
    if contrast > 0:
        transforms.append(lambda img: TF.adjust_contrast(img, 1 + torch.rand(1, generator=generator).item() * contrast))
    if saturation > 0:
        transforms.append(lambda img: TF.adjust_saturation(img, 1 + torch.rand(1, generator=generator).item() * saturation))
    if hue > 0:
        transforms.append(lambda img: TF.adjust_hue(img, torch.rand(1, generator=generator).item() * hue))
        
    if shuffle:
        transforms = torch.randperm(len(transforms), generator=generator).tolist()
        
    for t in transforms:
        if torch.rand(1, generator=generator).item() <= color_jitter_prob:
            image = t(image)
            
    if torch.rand(1, generator=generator).item() <= to_grayscale_prob:
        image = TF.rgb_to_grayscale(image, num_output_channels=3)
        
    return torch.clamp(image, 0.0, 1.0)


def random_flip(image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Randomly flip image horizontally."""
    if generator is None:
        generator = torch.Generator()
        
    if torch.rand(1, generator=generator).item() <= 0.5:
        return TF.hflip(image)
    return image


def gaussian_blur(
    image: torch.Tensor,
    blur_divider: float = 10.0,
    sigma_min: float = 0.1,
    sigma_max: float = 2.0,
    apply_prob: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply gaussian blur to image."""
    if generator is None:
        generator = torch.Generator()
        
    if torch.rand(1, generator=generator).item() > apply_prob:
        return image
        
    kernel_size = int(image.shape[0] / blur_divider)
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    sigma = sigma_min + torch.rand(1, generator=generator).item() * (sigma_max - sigma_min)
    return TF.gaussian_blur(image, kernel_size, [sigma, sigma])


def solarize(
    image: torch.Tensor,
    threshold: float,
    apply_prob: float = 1.0,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Apply solarization to image."""
    if generator is None:
        generator = torch.Generator()
        
    def _apply(img):
        return torch.where(img < threshold, img, 1.0 - img)
        
    return _maybe_apply(_apply, image, apply_prob, generator)
