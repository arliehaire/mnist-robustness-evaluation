import torch
import torchvision.transforms.functional as TF


def add_gaussian_noise(images, std):
    """
    Adds Gaussian noise to a batch of images.

    images shape: (B, 1, 28, 28)
    std: noise severity
    """
    noise = torch.randn_like(images) * std
    corrupted = images + noise
    return torch.clamp(corrupted, 0.0, 1.0)


def rotate_images(images, angle):
    """
    Rotates each image in a batch by a fixed angle.

    angle: degrees
    """
    rotated = []
    for img in images:
        rotated_img = TF.rotate(img, angle)
        rotated.append(rotated_img)

    return torch.stack(rotated)


def apply_occlusion(images, size):
    """
    Applies a black square occlusion to the center of each image.

    size: width/height of square
    """
    corrupted = images.clone()

    _, _, height, width = corrupted.shape

    top = (height - size) // 2
    left = (width - size) // 2

    corrupted[:, :, top:top + size, left:left + size] = 0.0

    return corrupted
