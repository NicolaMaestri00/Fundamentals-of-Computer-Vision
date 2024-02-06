import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.Dropout(0.1, inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.PReLU(out_channels)
    )


class Block(nn.Module):
    def __init__(self, in_channels, side_channels, out_channels, bias=True):
        super(Block, self).__init__()
        self.side_channels = side_channels
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = ConvBlock(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv3 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv4 = ConvBlock(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias)
        self.f = nn.Sequential(
            nn.Dropout(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.f(self.conv5(x + out))
        return out