"""
Train CycleGAN for Chest X-ray Image Translation.

Author: Vibha I S
"""

import os
import csv

import torch
import matplotlib.pyplot as plt
from PIL import Image
import IPython.display as display

from config import (
    DEVICE,
    DATASET_PATH,
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_DIR,
    GENERATED_IMAGES_DIR,
    LOSS_LOG_FILE
)

from models.generator import Generator
from models.discriminator import Discriminator

from utils.dataset import get_dataloaders
from utils.losses import criterion_GAN, criterion_cycle
from utils.visualize import visualize_cycle


# -----------------------------
# Model Initialization
# -----------------------------

generator_XtoY = Generator().to(DEVICE)
generator_YtoX = Generator().to(DEVICE)

discriminator_X = Discriminator().to(DEVICE)
discriminator_Y = Discriminator().to(DEVICE)


# -----------------------------
# Optimizers
# -----------------------------

optimizer_G = torch.optim.Adam(
    list(generator_XtoY.parameters()) +
    list(generator_YtoX.parameters()),
    lr=LEARNING_RATE
)

optimizer_D_X = torch.optim.Adam(
    discriminator_X.parameters(),
    lr=LEARNING_RATE
)

optimizer_D_Y = torch.optim.Adam(
    discriminator_Y.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training Step
# -----------------------------

def train_step(real_X, real_Y):

    real_X = real_X.to(DEVICE)
    real_Y = real_Y.to(DEVICE)

    fake_Y = generator_XtoY(real_X)
    fake_X = generator_YtoX(real_Y)

    reconstructed_X = generator_YtoX(fake_Y)
    reconstructed_Y = generator_XtoY(fake_X)

    loss_GAN_XY = criterion_GAN(
        discriminator_Y(fake_Y),
        torch.ones_like(discriminator_Y(fake_Y))
    )

    loss_GAN_YX = criterion_GAN(
        discriminator_X(fake_X),
        torch.ones_like(discriminator_X(fake_X))
    )

    loss_cycle = (
        criterion_cycle(reconstructed_X, real_X)
        +
        criterion_cycle(reconstructed_Y, real_Y)
    )

    generator_loss = (
        loss_GAN_XY
        +
        loss_GAN_YX
        +
        10 * loss_cycle
    )

    optimizer_G.zero_grad()
    generator_loss.backward()
    optimizer_G.step()

    discriminator_X_loss = 0.5 * (
        criterion_GAN(
            discriminator_X(real_X),
            torch.ones_like(discriminator_X(real_X))
        )
        +
        criterion_GAN(
            discriminator_X(fake_X.detach()),
            torch.zeros_like(discriminator_X(fake_X.detach()))
        )
    )

    optimizer_D_X.zero_grad()
    discriminator_X_loss.backward()
    optimizer_D_X.step()

    discriminator_Y_loss = 0.5 * (
        criterion_GAN(
            discriminator_Y(real_Y),
            torch.ones_like(discriminator_Y(real_Y))
        )
        +
        criterion_GAN(
            discriminator_Y(fake_Y.detach()),
            torch.zeros_like(discriminator_Y(fake_Y.detach()))
        )
    )

    optimizer_D_Y.zero_grad()
    discriminator_Y_loss.backward()
    optimizer_D_Y.step()

    return (
        generator_loss.item(),
        discriminator_X_loss.item(),
        discriminator_Y_loss.item()
    )
