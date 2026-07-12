"""
Dataset loader for Chest X-ray CycleGAN.
"""

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

from config import IMAGE_SIZE, BATCH_SIZE


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def get_dataloaders(dataset_path, batch_size=BATCH_SIZE):
    """
    Creates separate dataloaders for NORMAL and PNEUMONIA domains.
    """

    dataset = ImageFolder(
        root=dataset_path,
        transform=transform
    )

    normal_idx = dataset.class_to_idx["NORMAL"]
    pneumonia_idx = dataset.class_to_idx["PNEUMONIA"]

    normal_indices = [
        i for i in range(len(dataset))
        if dataset.imgs[i][1] == normal_idx
    ]

    pneumonia_indices = [
        i for i in range(len(dataset))
        if dataset.imgs[i][1] == pneumonia_idx
    ]

    normal_loader = DataLoader(
        Subset(dataset, normal_indices),
        batch_size=batch_size,
        shuffle=True
    )

    pneumonia_loader = DataLoader(
        Subset(dataset, pneumonia_indices),
        batch_size=batch_size,
        shuffle=True
    )

    return normal_loader, pneumonia_loader
