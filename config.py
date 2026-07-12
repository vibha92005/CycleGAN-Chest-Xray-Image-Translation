"""
Configuration file for CycleGAN training.
Modify the dataset path below if running on a different system.
"""

import torch

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Configuration
# Default Kaggle dataset path.
# Change this path if running locally or on another platform.
DATASET_PATH = "/kaggle/input/chest-xray-pneumonia/chest_xray/train"

# Image Parameters
IMAGE_SIZE = 128
CHANNELS = 1

# Training Parameters
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 3e-4

# Directories
CHECKPOINT_DIR = "checkpoints"
GENERATED_IMAGES_DIR = "generated_images"
LOSS_LOG_FILE = "loss_logs.csv"
