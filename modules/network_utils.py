import torch
import os
import tensorflow as tf
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


def load_pretrained_weights_from_tensorflow_to_pytorch(model):
    tf_path = os.path.abspath('./ckpt/model.ckpt-6935893')  # Path to our TensorFlow checkpoint
    init_vars = tf.train.list_variables(tf_path)
    tf_vars = []
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        tf_vars.append((name, array.squeeze()))

    for name, array in tf_vars:
        if not ('generator' in name or 'discriminator' in name):
            print(f'Excluded weights: {name} {array.shape}')
            continue

        if 'Adam' in name:
            print(f'Excluded weights: {name} {array.shape}')
            continue

        pointer = model
        if name.split('/')[-1] in ['u']:
            print(f'Excluded weights: {name}')
            continue
        name = name.split('/')
        for m_name in name:
            if m_name == 'kernel':
                pointer = getattr(pointer, 'weight')
                if len(pointer.shape) == 4:
                    # TODO: it might also be (3, 2, 0, 1), which needs double-checking
                    if len(array.shape) == 2:
                        # kernel size is 1x1
                        array = array[None, None]
                    elif len(array.shape) == 3:
                        array = array[..., None]
                    array = array.transpose(3, 2, 1, 0)
                elif len(pointer.shape) == 2:
                    array = array.transpose(1, 0)
            elif m_name == 'bias':
                pointer = getattr(pointer, 'bias')
                if len(array.shape) == 0:
                    array = array[None, ]
            else:
                pointer = getattr(pointer, m_name)

        try:
            assert pointer.shape == array.shape  # Catch error if the array shapes are not identical
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise

        # print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


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
