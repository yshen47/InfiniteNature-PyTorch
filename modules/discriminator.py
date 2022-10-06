import torch
from torch import nn
from modules.network_utils import sn_conv_padding, Conv2D
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = Conv2D(4, 64, kernel_size=4, stride=2, bias=True, use_spectrual_norm=False)
        self.conv_1 = Conv2D(64, 128, kernel_size=4, stride=1, bias=True, use_spectrual_norm=True)
        self.inst_norm_1 = torch.nn.InstanceNorm2d(128)

        self.conv_2 = Conv2D(128, 256, kernel_size=4, stride=1, bias=True, use_spectrual_norm=True)
        self.inst_norm_2 = torch.nn.InstanceNorm2d(256)

        self.conv_3 = Conv2D(256, 512, kernel_size=4, stride=2, bias=True, use_spectrual_norm=True)
        self.inst_norm_3 = torch.nn.InstanceNorm2d(512)
        self.D_logit = Conv2D(512, 1, kernel_size=4, stride=1, bias=True, use_spectrual_norm=False)

    def forward(self, rgbd_sequence):
        rgbd_sequence = sn_conv_padding(rgbd_sequence, stride=2, kernel_size=4)
        x = self.conv(rgbd_sequence)

        features = []
        x1 = sn_conv_padding(x, stride=1, kernel_size=4)
        x1 = self.conv_1(x1)
        x1 = self.inst_norm_1(x1)
        x1 = F.leaky_relu(x1, 0.2)
        features.append(x1)

        x2 = sn_conv_padding(x1, stride=1, kernel_size=4)
        x2 = self.conv_2(x2)
        x2 = self.inst_norm_2(x2)
        x2 = F.leaky_relu(x2, 0.2)
        features.append(x2)

        x3 = sn_conv_padding(x2, stride=2, kernel_size=4)
        x3 = self.conv_3(x3)
        x3 = self.inst_norm_3(x3)
        x3 = F.leaky_relu(x3, 0.2)
        features.append(x3)

        logit = self.D_logit(x3)
        return features, logit