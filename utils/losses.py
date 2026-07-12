"""
Loss functions used for CycleGAN.
"""

import torch.nn as nn

criterion_GAN = nn.MSELoss()

criterion_cycle = nn.L1Loss()
