"""
model.py
--------
A compact U-Net for multi-class semantic segmentation.
"""

import torch
import torch.nn as nn

import config


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with ReLU + dropout, same as each notebook block."""

    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=None, num_classes=None, base_filters=None, dropout=None):
        super().__init__()
        in_channels = in_channels or config.IN_CHANNELS
        num_classes = num_classes or config.NUM_CLASSES
        f = base_filters or config.BASE_FILTERS
        dropout = config.DROPOUT if dropout is None else dropout

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, f, dropout)
        self.enc2 = ConvBlock(f, f * 2, dropout)
        self.enc3 = ConvBlock(f * 2, f * 4, dropout)
        self.enc4 = ConvBlock(f * 4, f * 8, dropout)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16, dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8, dropout)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4, dropout)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2, dropout)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f, dropout)

        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)   # raw logits, shape (N, num_classes, H, W)
