import torch
import math
import torch.nn.functional as F
import math


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
