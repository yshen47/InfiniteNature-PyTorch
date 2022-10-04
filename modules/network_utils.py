import torch.nn.functional as F
import math
from torch import nn

def sn_conv_padding(x, stride, kernel_size, mode='reflect'):
    h, w = x.shape[2:]
    output_h, output_w = int(math.ceil(h / stride)), int(
        math.ceil(w / stride))

    p_h = output_h * stride + kernel_size - h - 1
    p_w = output_w * stride + kernel_size - w - 1

    pad_top = p_h // 2
    pad_bottom = p_h - pad_top
    pad_left = p_w // 2
    pad_right = p_w - pad_left
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
    return x

class Dense(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.dense = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return self.dense(x)


class Conv2D(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride, bias, use_spectrual_norm):
        super().__init__()
        self.conv2d = nn.Conv2d(input_channel, output_channel,
                                kernel_size=kernel_size,
                                stride=stride,
                                bias=bias)
        if use_spectrual_norm:
            self.conv2d = nn.utils.spectral_norm(self.conv2d)

    def forward(self, x):
        return self.conv2d(x)
