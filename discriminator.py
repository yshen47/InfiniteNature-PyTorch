import torch
from torch import nn
from network_utils import sn_conv_padding
import torch.nn.functional as F


class MultiscaleDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.spade_discriminator_0 = PatchDiscriminator()
        self.spade_discriminator_1 = PatchDiscriminator()

    def forward(self, rgbd_sequence):
        features, logit = self.spade_discriminator_0(rgbd_sequence)

        # half size by averaging each four pixels
        x_small = (rgbd_sequence[:, :, 0::2, 0::2] + rgbd_sequence[:, :, 0::2, 1::2]
               + rgbd_sequence[:, :, 1::2, 0::2] + rgbd_sequence[:, :, 1::2, 1::2]) / 4

        features_small, logit_small = self.spade_discriminator_1(x_small)
        return [features, features_small], [logit, logit_small]


class PatchDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 64, kernel_size=4, stride=2, bias=True)
        self.conv_1 = nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=1, bias=True))
        self.inst_norm_1 = torch.nn.InstanceNorm2d(64)

        self.conv_2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=1, bias=True))
        self.inst_norm_2 = torch.nn.InstanceNorm2d(128)

        self.conv_3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=True))
        self.inst_norm_3 = torch.nn.InstanceNorm2d(256)
        self.D_logit = nn.Conv2d(256, 1, kernel_size=4, stride=1, bias=True)


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