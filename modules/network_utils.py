import torch.nn.functional as F
import math
from torch import nn
import tensorflow as tf
import torch


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


class DummyClass:
    def __init__(self, shape):
        self.shape = shape


class Conv2D(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride, bias, use_spectrual_norm):
        super().__init__()
        self.conv2d = nn.Conv2d(input_channel, output_channel,
                                kernel_size=kernel_size,
                                stride=stride,
                                bias=bias)
        self._u = None # shape is randomly fed as placeholder
        self._bias = None # DummyClass(self.conv2d.bias.shape) if bias else None
        self._weight = None #DummyClass(self.conv2d.weight.shape)
        self.use_spectral_norm = use_spectrual_norm
        if self.use_spectral_norm:
            self.applied = False

    @property
    def u(self):
        return self._u

    @property
    def bias(self):
        return self._bias

    @property
    def weight(self):
        return self._weight

    @property
    def shape(self):
        return self.conv2d.weight.shape

    @u.setter
    def u(self, new_value):
        self._u = new_value
        if self.use_spectral_norm and not self.applied:
            self.apply_spectral_norm_to_kernel()

    @bias.setter
    def bias(self, new_value):
        print('entered bias.setter')
        self._bias = new_value
        self.conv2d.bias.data = new_value

    @weight.setter
    def weight(self, new_value):
        print('entered weight.setter')
        self.conv2d.weight.data = new_value
        if self.use_spectral_norm and not self.applied:
            self.apply_spectral_norm_to_kernel()
        self._weight = self.conv2d.weight.data

    def forward(self, x):
        return self.conv2d(x)

    def apply_spectral_norm_to_kernel(self, iteration=1):
        """Applies spectral normalization to a weight tensor. from original infinite nature codebase

        When update_variable is True, updates the u vector of spectral normalization
        with its power-iteration method. If spectral norm is called multiple
        times within the same scope (like in Infinite Nature), the normalization
        variable u will be shared between them, and any prior assign operations on u
        will be executed before the current assign. Because power iteration is
        convergent, it does not matter if multiple updates take place in a single
        forward pass.

        Args:
          w: (tensor) A weight tensor to apply spectral normalization to
          iteration: (int) The number of times to run power iteration when called

        Returns:
          A tensor of the same shape as w.
        """
        if self._u is None or self._weight is None:
            return

        w = tf.constant(self.conv2d.weight.data.permute(2, 3, 1, 0).detach().cpu(), dtype=tf.float32)
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u_hat = tf.constant(self.u.data.cpu()[None, ], dtype=tf.float32)

        v_hat = None
        for _ in range(iteration):
            # Power iteration. Usually iteration = 1 will be enough.
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
        w_norm = torch.from_numpy(w_norm.numpy()).permute(3, 2, 0, 1).to(self.conv2d.weight.data.device)
        self.conv2d.weight.data = w_norm
        self.applied = True