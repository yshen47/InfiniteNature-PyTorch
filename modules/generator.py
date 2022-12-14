import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
from modules.network_utils import sn_conv_padding, Dense, Conv2D


class Generator(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.spade_encoder = SpadeEncoder(args)
        self.spade_network_noenc = SpadeNetworkNoEnc(args)

    def forward(self, rgbd, mask, z):
        return self.spade_network_noenc(rgbd, mask, z)

    def style_encoding(self, encoding, return_mulogvar=False, sample=False):
        mu, logvar = self.spade_encoder(encoding)
        if sample:
            z = self.spade_encoder.reparameterize(mu, logvar)
        else:
            z = mu
        if return_mulogvar:
            return z, mu, logvar
        return z


class SpadeNetworkNoEnc(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.spade_generator = SpadeGenerator(args)

    def forward(self, rgbd, mask, z):
        return self.spade_generator(rgbd, mask, z)


class SpadeGenerator(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.dataset == 'google_earth':
            self.init_h = 8
            self.init_w = 8
        elif args.dataset == 'clevr-infinite':
            self.init_h = 8
            self.init_w = 8
        else:
            self.init_h = 5
            self.init_w = 8
        self.linear = Dense(self.args.embedding_size, 16 * self.args.num_channel * self.init_h * self.init_w)
        self.head = SpadeResBlock(args,
                                  channel_in=16 * self.args.num_channel,
                                  channel_out=16 * self.args.num_channel,
                                  use_spectral_norm=self.args.use_spectral_norm,
                                  in_channel=5)

        self.middle_0 = SpadeResBlock(args,
                                      channel_in=16 * self.args.num_channel,
                                      channel_out=16 * self.args.num_channel,
                                      use_spectral_norm=self.args.use_spectral_norm,
                                      in_channel=5)

        self.middle_1 = SpadeResBlock(args,
                                      channel_in=16 * self.args.num_channel,
                                      channel_out=16 * self.args.num_channel,
                                      use_spectral_norm=self.args.use_spectral_norm,
                                      in_channel=5)

        self.up_0 = SpadeResBlock(args,
                                  channel_in=16 * self.args.num_channel,
                                  channel_out=8 * self.args.num_channel,
                                  use_spectral_norm=self.args.use_spectral_norm,
                                  in_channel=5)

        self.up_1 = SpadeResBlock(args,
                                  channel_in=8 * self.args.num_channel,
                                  channel_out=4 * self.args.num_channel,
                                  use_spectral_norm=self.args.use_spectral_norm,
                                  in_channel=5)

        self.up_2 = SpadeResBlock(args,
                                  channel_in=4 * self.args.num_channel,
                                  channel_out=2 * self.args.num_channel,
                                  use_spectral_norm=self.args.use_spectral_norm,
                                  in_channel=5)

        self.up_3 = SpadeResBlock(args,
                                  channel_in=2 * self.args.num_channel,
                                  channel_out=self.args.num_channel,
                                  use_spectral_norm=self.args.use_spectral_norm,
                                  in_channel=5)
        if args.dataset == 'google_earth':
            self.up_4 = SpadeResBlock(args,
                                      channel_in=self.args.num_channel,
                                      channel_out=self.args.num_channel,
                                      use_spectral_norm=self.args.use_spectral_norm,
                                      in_channel=5)

        self.conv = Conv2D(self.args.num_channel, 4, kernel_size=3, stride=1, bias=True, use_spectrual_norm=True)

    def forward(self, rgbd, mask, z):
        """

        :param rgbd: [B, 4, H, W] the rendered view to be refined
        :param mask: [B, 1, H, W] binary mask of unknown regions. 1 where known and 0 where
      unknown
        :param z: [B, D] a noise vector to be used as noise for the generator
        :return: [B, 4, H, W] refined rgbd image.
        """
        img = 2 * rgbd - 1
        img = torch.cat([img, mask], dim=1)

        batch_size, unused_c, im_height, im_width = rgbd.shape
        x = self.linear(z).view(batch_size, self.init_h, self.init_w, 16 * self.args.num_channel).permute(0, 3, 1, 2)

        x = self.head(x, img)
        x = F.interpolate(x, scale_factor=2)
        x = self.middle_0(x, img)
        x = self.middle_1(x, img)
        x = F.interpolate(x, scale_factor=2)
        x = self.up_0(x, img)
        x = F.interpolate(x, scale_factor=2)

        x = self.up_1(x, img)
        x = F.interpolate(x, scale_factor=2)

        x = self.up_2(x, img)
        x = F.interpolate(x, scale_factor=2)

        x = self.up_3(x, img)
        if self.args.dataset == 'google_earth':
            x = F.interpolate(x, scale_factor=2)
            x = self.up_4(x, img)

        x = F.leaky_relu(x, 0.2)
        x = sn_conv_padding(x, stride=1, kernel_size=3)
        x = self.conv(x)
        x = torch.tanh(x)
        return 0.5 * (x + 1)


class SpadeEncoder(nn.Module):

    def __init__(self, args, num_channel=16):
        super().__init__()
        self.args = args
        self.num_channel = num_channel
        self.conv_0 = Conv2D(4, num_channel, kernel_size=3, stride=2, bias=True, use_spectrual_norm=True)
        self.inst_norm_0 = torch.nn.InstanceNorm2d(num_channel)

        self.conv_1 = Conv2D(num_channel, 2 * num_channel, kernel_size=3, stride=2, bias=True, use_spectrual_norm=True)
        self.inst_norm_1 = torch.nn.InstanceNorm2d(2 * num_channel)

        self.conv_2 = Conv2D(2 * num_channel, 4 * num_channel, kernel_size=3, stride=2, bias=True,
                             use_spectrual_norm=True)
        self.inst_norm_2 = torch.nn.InstanceNorm2d(4 * num_channel)

        self.conv_3 = Conv2D(4 * num_channel, 8 * num_channel, kernel_size=3, stride=2, bias=True,
                             use_spectrual_norm=True)

        self.inst_norm_3 = torch.nn.InstanceNorm2d(8 * num_channel)

        self.conv_4 = Conv2D(8 * num_channel, 8 * num_channel, kernel_size=3, stride=2, bias=True,
                             use_spectrual_norm=True)
        self.inst_norm_4 = torch.nn.InstanceNorm2d(8 * num_channel)

        self.conv_5 = Conv2D(8 * num_channel, 8 * num_channel, kernel_size=3, stride=2, bias=True,
                             use_spectrual_norm=True)
        self.inst_norm_5 = torch.nn.InstanceNorm2d(8 * num_channel)
        if args.dataset == 'google_earth':
            self.conv_6 = Conv2D(8 * num_channel, 8 * num_channel, kernel_size=3, stride=2, bias=True,
                                 use_spectrual_norm=True)
            self.inst_norm_6 = torch.nn.InstanceNorm2d(8 * num_channel)
        if args.dataset == 'google_earth':
            input_dense_dim = 2048 # 8192
        elif args.dataset == 'clevr-infinite':
            input_dense_dim = 2048
        else:
            input_dense_dim = 1536

        self.linear_mu = Dense(input_dense_dim, self.args.embedding_size)
        self.linear_logvar = Dense(input_dense_dim, self.args.embedding_size)

    def forward(self, x, return_intermediate=False):
        """Encoder that outputs global N(mu, sig) parameters.
          Args:
            x: [B, 4, H, W an RGBD image (usually the initial image) which is used to
              sample noise from a distirbution to feed into the refinement
              network. Range [0, 1].
            scope: (str) variable scope
          Returns:
            (mu, logvar) are [B, 256] tensors of parameters defining a normal
              distribution to sample from.
        """
        x = 2 * x - 1

        x1 = sn_conv_padding(x, stride=2, kernel_size=3)
        x1 = self.conv_0(x1)
        x1 = self.inst_norm_0(x1)
        x1 = F.leaky_relu(x1, 0.2)

        x2 = sn_conv_padding(x1, stride=2, kernel_size=3)
        x2 = self.conv_1(x2)
        x2 = self.inst_norm_1(x2)
        x2 = F.leaky_relu(x2, 0.2)

        x3 = sn_conv_padding(x2, stride=2, kernel_size=3)
        x3 = self.conv_2(x3)
        x3 = self.inst_norm_2(x3)
        x3 = F.leaky_relu(x3, 0.2)

        x4 = sn_conv_padding(x3, stride=2, kernel_size=3)
        x4 = self.conv_3(x4)
        x4 = self.inst_norm_3(x4)
        x4 = F.leaky_relu(x4, 0.2)

        x5 = sn_conv_padding(x4, stride=2, kernel_size=3)
        x5 = self.conv_4(x5)
        x5 = self.inst_norm_4(x5)
        x5 = F.leaky_relu(x5, 0.2)

        x6 = sn_conv_padding(x5, stride=2, kernel_size=3)
        x6 = self.conv_5(x6)
        x = self.inst_norm_5(x6)
        if self.args.dataset == 'google_earth':
            x6 = F.leaky_relu(x, 0.2)
            x7 = sn_conv_padding(x6, stride=2, kernel_size=3)
            x7 = self.conv_5(x7)
            x = self.inst_norm_6(x7)

        features = F.leaky_relu(x, 0.2)
        bs, feature_dim, s1, s2 = features.shape
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(bs, -1)
        mu = self.linear_mu(features)
        logvar = self.linear_logvar(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """

        :param mu: Mean of normal noise to sample
        :param logvar: log variance of normal noise to sample
        :return: Random Gaussian sampled from mu and logvar.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu


# Modified based on infinite nature spade block
class SpadeResBlock(nn.Module):

    def __init__(self, args, channel_in, channel_out, use_spectral_norm=False, in_channel=5):
        super().__init__()
        self.args = args
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.use_spectral_norm = use_spectral_norm
        channel_middle = min(channel_in, channel_out)

        self.spade_0 = Spade(args, channel_in, in_channel=in_channel)
        self.conv_0 = Conv2D(channel_in, channel_middle, kernel_size=3, stride=1, bias=True, use_spectrual_norm=True)

        self.spade_1 = Spade(args, channel_middle, in_channel=in_channel)
        self.conv_1 = Conv2D(channel_middle, channel_middle, kernel_size=3, stride=1, bias=True,
                             use_spectrual_norm=True)

        if self.channel_in != self.channel_out:
            self.shortcut_spade = Spade(args, channel_in, in_channel=in_channel)
            self.shortcut_conv = Conv2D(channel_in, channel_out, kernel_size=1, stride=1, bias=False,
                                        use_spectrual_norm=True)

    def forward(self, tensor, condition):
        x = F.leaky_relu(self.spade_0(tensor, condition), 0.2)
        x = sn_conv_padding(x, stride=1, kernel_size=3)
        x = self.conv_0(x)

        x = F.leaky_relu(self.spade_1(x, condition), 0.2)
        x = sn_conv_padding(x, stride=1, kernel_size=3)
        x = self.conv_1(x)

        if self.channel_in != self.channel_out:
            x_in = self.shortcut_spade(tensor, condition)
            x_in = sn_conv_padding(x_in, stride=1, kernel_size=1)
            x_in = self.shortcut_conv(x_in)
        else:
            x_in = tensor
        out = x + x_in
        return out


class Spade(nn.Module):

    def __init__(self, args, channel_size, in_channel=5):
        super().__init__()
        self.args = args
        self.num_hidden = self.args.num_hidden
        self.channel_size = channel_size
        self.instance_norm = torch.nn.InstanceNorm2d(channel_size)
        self.conv_cond = Conv2D(in_channel, self.num_hidden, kernel_size=3, stride=1, bias=True,
                                use_spectrual_norm=True)

        self.gamma = Conv2D(self.num_hidden, channel_size, kernel_size=3, stride=1, bias=True, use_spectrual_norm=False)
        self.beta = Conv2D(self.num_hidden, channel_size, kernel_size=3, stride=1, bias=True, use_spectrual_norm=False)

    def forward(self, x, condition):
        h, w = x.shape[-2:]
        x_normed = self.instance_norm(x)

        condition = self.diff_resize_area(condition, (h, w))
        condition = sn_conv_padding(condition, kernel_size=3, stride=1)
        condition = F.relu(self.conv_cond(condition))

        condition = sn_conv_padding(condition, stride=1, kernel_size=3, mode='constant')
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        out = x_normed * (1 + gamma) + beta
        return out

    def diff_resize_area(self, tensor, new_height_width):

        new_h, new_w = new_height_width
        unused_b, unused_d, curr_h, curr_w = tensor.shape

        # The least common multiplier used to determine the intermediate resize
        # operation.
        l_h = np.lcm(curr_h, new_h)
        l_w = np.lcm(curr_w, new_w)
        if l_h == curr_h and l_w == curr_w:
            im = tensor
        elif (l_h < (10 * new_h) and l_w < (10 * new_w)):
            # https://github.com/pytorch/pytorch/issues/10604
            im = torch.nn.functional.interpolate(
                tensor, [l_h, l_w], align_corners=False, mode='bilinear')
        else:
            raise RuntimeError("DifferentiableResizeArea is memory inefficient"
                               "for resizing from (%d, %d) -> (%d, %d)" %
                               (curr_h, curr_w, new_h, new_w))
        lh_factor = l_h // new_h
        lw_factor = l_w // new_w
        if lh_factor == lw_factor == 1:
            return im
        return F.avg_pool2d(
            im, [lh_factor, lw_factor], [lh_factor, lw_factor], padding=0)
