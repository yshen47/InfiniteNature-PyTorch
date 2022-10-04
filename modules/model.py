from modules.generator import Generator
from modules.discriminator import PatchDiscriminator
import torch
import pytorch_lightning as pl
from modules.warp import render_projection_from_srcs_fast
from modules.lpips import LPIPS
import os
from modules.loss import *


class InfiniteNature(pl.LightningModule):

    def __init__(self, generator_config, loss_config, data_config, learning_rate, ckpt_path=None, ignore_keys=()):
        super().__init__()
        self.generator_config = generator_config
        self.loss_config = loss_config
        self.data_config = data_config
        self.learning_rate = learning_rate
        self.generator = Generator(generator_config)
        self.spade_discriminator_0 = PatchDiscriminator()
        self.spade_discriminator_1 = PatchDiscriminator()
        self.perceptual_loss = LPIPS().eval()
        self.automatic_optimization = False
        if ckpt_path is None:
            self.load_pretrained_weights_from_tensorflow_to_pytorch()
        else:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.spade_discriminator_0.parameters()) +
                                    list(self.spade_discriminator_1.parameters()),
                                lr=lr, betas=(0.5, 0.9))
        return opt_g, opt_disc

    def init_from_ckpt(self, path, ignore_keys=('loss')):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                #print(ik, k)
                if k.startswith(ik):
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def load_pretrained_weights_from_tensorflow_to_pytorch(self):
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

            pointer = self
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
                        array = array[None,]
                else:
                    pointer = getattr(pointer, m_name)

            try:
                assert pointer.shape == array.shape  # Catch error if the array shapes are not identical
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise

            # print("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)

    def training_step(self, batch, batch_idx):
        x_src = torch.cat([batch['src_img'],
                           batch['src_depth']], dim=-1).permute(0, 3, 1, 2)
        gt_tgt_rgbd = torch.cat([batch['dst_img'],
                           batch['dst_depth']], dim=-1).permute(0, 3, 1, 2)
        z, mu, logvar = self.generator.style_encoding(x_src, return_mulogvar=True)
        rendered_rgbd, mask = self.render_with_projection(x_src[:, :3][:, None],
                                                          x_src[:, 3][:, None],
                                                          batch["Ks"][:, 0],
                                                          batch["Ks"][:, 0],
                                                          batch['T_src2tgt'])
        predicted_rgbd = self(rendered_rgbd, mask, z)
        # disc_on_generated = self.discriminate(predicted_rgbd)
        # generated_features = [f[0] for f in disc_on_generated]
        # generated_logits = [f[1] for f in disc_on_generated]
        loss_dict = compute_infinite_nature_loss(predicted_rgbd, gt_tgt_rgbd, self.discriminate, (mu, logvar), self.perceptual_loss)
        self.log_dict(loss_dict)

        opt_ae, opt_disc = self.optimizers()
        opt_ae.zero_grad()
        loss_dict['total_generator_loss'].backward()
        opt_ae.step()
        opt_disc.zero_grad()
        loss_dict['total_discriminator_loss'].backward()
        opt_disc.step()

    def validation_step(self, batch, batch_idx):
        x_src = torch.cat([batch['src_img'],
                           batch['src_depth']], dim=-1).permute(0, 3, 1, 2)
        gt_tgt_rgbd = torch.cat([batch['dst_img'],
                           batch['dst_depth']], dim=-1).permute(0, 3, 1, 2)

        z, mu, logvar = self.generator.style_encoding(x_src, return_mulogvar=True)
        rendered_rgbd, mask = self.render_with_projection(x_src[:, :3][:, None],
                                                          x_src[:, 3][:, None],
                                                          batch["Ks"][:, 0],
                                                          batch["Ks"][:, 0],
                                                          batch['T_src2tgt'])
        predicted_rgbd = self(rendered_rgbd, mask, z)
        loss_dict = compute_infinite_nature_loss(predicted_rgbd, gt_tgt_rgbd, self.discriminate, (mu, logvar), self.perceptual_loss)

    def render_with_projection(self, x_src, dm_src, K_src, K_next, T_src2tgt):
        warped_depth, warped_features, extrapolation_mask = render_projection_from_srcs_fast(
            x_src,
            dm_src,
            K_next.to(self.device),
            K_src.to(self.device),
            T_src2tgt,
            src_num=x_src.shape[1])
        warped_rgbd = torch.cat([warped_features, warped_depth], dim=1)
        return warped_rgbd, 1-extrapolation_mask

    def forward(self, rendered_rgbd, mask, encoding):
        predicted_rgbd = self.generator(rendered_rgbd, mask, encoding)
        refined_disparity = self.rescale_refined_disparity(rendered_rgbd[:, 3:], mask, predicted_rgbd[:, 3:])
        predicted_rgbd = torch.cat([predicted_rgbd[:, :3], refined_disparity], axis=1)
        return predicted_rgbd

    def discriminate(self, rgbd, use_for_discriminator_loss=False):
        if use_for_discriminator_loss:
            rgbd = rgbd.detach()
        else:
            self.spade_discriminator_0.eval()
            self.spade_discriminator_1.eval()
        features, logit = self.spade_discriminator_0(rgbd)

        # half size by averaging each four pixels
        x_small = (rgbd[:, :, 0::2, 0::2] + rgbd[:, :, 0::2, 1::2]
                   + rgbd[:, :, 1::2, 0::2] + rgbd[:, :, 1::2, 1::2]) / 4

        features_small, logit_small = self.spade_discriminator_1(x_small)

        if not use_for_discriminator_loss:
            self.spade_discriminator_0.train()
            self.spade_discriminator_1.train()
        return [features, features_small], [logit, logit_small]

    def rescale_refined_disparity(self, rendered_disparity, input_mask, refined_disparity):
        """Rescales the refined disparity to match the input's scale.
        This is done to prevent drifting in the disparity. We match the scale
        by solving a least squares optimization.
        Args:
          rendered_disparity: [B, H, W, 1] disparity produced by the render step
          input_mask: [B, H, W, 1] a mask with 1's denoting regions that were
            visible through the rendering.
          refined_disparity: [B, H, W, 1] disparity of the refinement network output
        Returns:
          refined_disparity that has been scale and shifted to match the statistics
          of rendered_disparity.
        """
        log_refined = torch.log(torch.clip(refined_disparity, 1e-2, 1))
        log_rendered = torch.log(torch.clip(rendered_disparity, 1e-2, 1))
        log_scale = torch.sum(input_mask * (log_rendered - log_refined)) / (
                torch.sum(input_mask) + 1e-4)
        scale = torch.exp(log_scale)
        scaled_refined_disparity = torch.clip(scale * refined_disparity,
                                                    0, 1)
        return scaled_refined_disparity
