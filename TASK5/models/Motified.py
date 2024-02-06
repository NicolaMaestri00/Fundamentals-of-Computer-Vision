import torch
import torch.nn as nn
import torch.nn.functional as F
from .CSP import *

class Net(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10, img_size = 100, bias=True):
        super(Net, self).__init__()
        self.model = 'CSP_mod5'
        self.inconv = ConvBlock(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=bias)
        self.block1 = Block(32, 16, 64)
        self.block2 = Block(64, 32, 128)
        self.block3 = Block(128, 64, 128)
        self.block4 = Block(128, 64, 128)
        self.fc = nn.Sequential(
            nn.Linear(pow(img_size//16+1, 2)*128, num_classes),
            nn.Dropout(0.1, inplace=True),
        )

    def forward(self, x):
        out = self.inconv(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        # out = torch.nn.functional.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out