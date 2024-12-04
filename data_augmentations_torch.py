from functools import partial
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def random_crop(img: torch.Tensor, rng: torch.Generator, padding: int) -> torch.Tensor:
    """Apply random crop with padding"""
    crop_from = torch.randint(0, 2 * padding + 1, (2,), generator=rng, device=img.device)
    crop_from = torch.cat([crop_from, torch.zeros(1, dtype=torch.int64, device=img.device)])
    
    padded_img = F.pad(
        img.permute(2, 0, 1),  # Convert to [C, H, W] for padding
        (padding, padding, padding, padding),
        mode="replicate"
    ).permute(1, 2, 0)  # Back to [H, W, C]
    
    return padded_img[
        crop_from[0]:crop_from[0] + img.shape[0],
        crop_from[1]:crop_from[1] + img.shape[1],
        :
    ]

def batched_random_crop(img: torch.Tensor, rng: torch.Generator, padding: int, num_batch_dims: int = 1) -> torch.Tensor:
    """Apply random crop to batched images"""
    # Flatten batch dims
    original_shape = img.shape
    img = img.reshape(-1, *img.shape[num_batch_dims:])
    
    # Create separate generators for each image
    rngs = [torch.Generator(device=img.device).manual_seed(int(torch.randint(1<<31, (1,), generator=rng).item()))
            for _ in range(img.shape[0])]
    
    # Apply random crop to each image
    crops = []
    for single_img, single_rng in zip(img, rngs):
        crops.append(random_crop(single_img, single_rng, padding))
    img = torch.stack(crops)
    
    # Restore batch dims
    return img.reshape(original_shape)

def resize(image: torch.Tensor, image_dim: tuple) -> torch.Tensor:
    """Resize image to given dimensions"""
    assert len(image_dim) == 2
    return F.interpolate(
        image.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
        size=image_dim,
        mode='bilinear',
        align_corners=False
    ).squeeze(0).permute(1, 2, 0)  # Back to [H, W, C]

def _maybe_apply(apply_fn, inputs: torch.Tensor, rng: torch.Generator, apply_prob: float) -> torch.Tensor:
    """Conditionally apply function with probability"""
    should_apply = torch.rand(1, generator=rng, device=inputs.device).item() <= apply_prob
    return apply_fn(inputs) if should_apply else inputs

def rgb_to_hsv(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> tuple:
    """Convert RGB to HSV color space"""
    return TF.rgb_to_hsv(torch.stack([r, g, b], dim=-1))

def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> tuple:
    """Convert HSV to RGB color space"""
    hsv = torch.stack([h, s, v], dim=-1)
    rgb = TF.hsv_to_rgb(hsv)
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]

def adjust_brightness(rgb_tuple: tuple, delta: float) -> tuple:
    """Adjust brightness of RGB image"""
    return tuple(x + delta for x in rgb_tuple)

def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust contrast of image"""
    mean = image.mean(dim=(-2, -1), keepdim=True)
    return factor * (image - mean) + mean

def adjust_saturation(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, factor: float) -> tuple:
    """Adjust saturation in HSV space"""
    return h, torch.clamp(s * factor, 0.0, 1.0), v

def adjust_hue(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, delta: float) -> tuple:
    """Adjust hue in HSV space"""
    return (h + delta) % 1.0, s, v

def color_transform(
    image: torch.Tensor,
    rng: torch.Generator,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    to_grayscale_prob: float = 0.0,
    color_jitter_prob: float = 1.0,
    apply_prob: float = 1.0,
    shuffle: bool = True
) -> torch.Tensor:
    """Apply color jittering to image"""
    def _to_grayscale(image):
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device)
        grayscale = (image * rgb_weights).sum(dim=-1, keepdim=True)
        return grayscale.repeat(1, 1, 3)

    should_apply = torch.rand(1, generator=rng, device=image.device).item() <= apply_prob
    should_apply_gs = torch.rand(1, generator=rng, device=image.device).item() <= to_grayscale_prob
    should_apply_color = torch.rand(1, generator=rng, device=image.device).item() <= color_jitter_prob

    if should_apply and should_apply_color:
        transforms = []
        if brightness > 0:
            transforms.append(lambda img: TF.adjust_brightness(img, 1 + torch.rand(1, generator=rng).item() * brightness))
        if contrast > 0:
            transforms.append(lambda img: TF.adjust_contrast(img, 1 + torch.rand(1, generator=rng).item() * contrast))
        if saturation > 0:
            transforms.append(lambda img: TF.adjust_saturation(img, 1 + torch.rand(1, generator=rng).item() * saturation))
        if hue > 0:
            transforms.append(lambda img: TF.adjust_hue(img, torch.rand(1, generator=rng).item() * hue))
            
        if shuffle:
            indices = torch.randperm(len(transforms), generator=rng)
            transforms = [transforms[i] for i in indices]
            
        image_NCHW = image.permute(2, 0, 1).unsqueeze(0)
        for t in transforms:
            image_NCHW = t(image_NCHW)
        image = image_NCHW.squeeze(0).permute(1, 2, 0)
        
    if should_apply and should_apply_gs:
        image = _to_grayscale(image)
        
    return torch.clamp(image, 0.0, 1.0)

def gaussian_blur(
    image: torch.Tensor,
    rng: torch.Generator,
    blur_divider: float = 10.0,
    sigma_min: float = 0.1,
    sigma_max: float = 2.0,
    apply_prob: float = 1.0
) -> torch.Tensor:
    """Apply gaussian blur to image"""
    kernel_size = int(image.shape[0] / blur_divider) | 1  # Ensure odd kernel size
    
    def _apply(image):
        sigma = sigma_min + torch.rand(1, generator=rng, device=image.device).item() * (sigma_max - sigma_min)
        return TF.gaussian_blur(
            image.permute(2, 0, 1).unsqueeze(0),
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma]
        ).squeeze(0).permute(1, 2, 0)
        
    return _maybe_apply(_apply, image, rng, apply_prob)

def solarize(image: torch.Tensor, rng: torch.Generator, threshold: float, apply_prob: float) -> torch.Tensor:
    """Apply solarize effect to image"""
    def _apply(image):
        return torch.where(image < threshold, image, 1.0 - image)
    return _maybe_apply(_apply, image, rng, apply_prob) 