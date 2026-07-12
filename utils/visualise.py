"""
Visualization utilities for CycleGAN.
"""

import os

import torch
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

from config import DEVICE


def visualize_cycle(
    epoch,
    loader,
    generator,
    reverse_generator,
    domain_name,
    save_dir="generated_images"
):
    """
    Saves real, translated and reconstructed images.
    """

    generator.eval()
    reverse_generator.eval()

    images, _ = next(iter(loader))

    images = images[:4].to(DEVICE)

    with torch.no_grad():

        fake_images = generator(images)

        reconstructed_images = reverse_generator(fake_images)

    def denormalize(image):
        return image * 0.5 + 0.5

    images = denormalize(images).cpu()

    fake_images = denormalize(fake_images).cpu()

    reconstructed_images = denormalize(reconstructed_images).cpu()

    real_grid = make_grid(images, nrow=4, padding=2)

    fake_grid = make_grid(fake_images, nrow=4, padding=2)

    reconstruction_grid = make_grid(
        reconstructed_images,
        nrow=4,
        padding=2
    )

    final_grid = torch.cat(
        [
            real_grid,
            fake_grid,
            reconstruction_grid
        ],
        dim=1
    )

    plt.figure(figsize=(12, 6))

    plt.imshow(
        final_grid.permute(1, 2, 0).squeeze(),
        cmap="gray"
    )

    plt.axis("off")

    plt.title(
        f"{domain_name} Domain | Epoch {epoch}\n"
        "Top: Real | Middle: Translated | Bottom: Reconstructed"
    )

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(
            save_dir,
            f"cycle_{domain_name}_epoch_{epoch}.png"
        )
    )

    plt.close()

    generator.train()

    reverse_generator.train()
