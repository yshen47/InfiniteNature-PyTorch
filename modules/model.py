from modules.generator import Generator
from modules.discriminator import PatchDiscriminator
import pytorch_lightning as pl
from modules.warp import render_projection_from_srcs_fast
from modules.lpips import LPIPS
import torch.nn.functional as F
from modules.loss import *
from modules.network_utils import Conv2D
import PIL
from modules.metrics import *
import os


class InfiniteNature(pl.LightningModule):

    def __init__(self, generator_config, learning_rate, ckpt_path=None, ignore_keys=()):
        super().__init__()
        self.generator_config = generator_config
        self.learning_rate = learning_rate
        self.generator = Generator(generator_config)
        self.perceptual_loss = LPIPS().eval()
        self.spade_discriminator_0 = PatchDiscriminator()
        self.spade_discriminator_1 = PatchDiscriminator()
        self.perceptual_loss = LPIPS().eval()
        self.automatic_optimization = False
        if ckpt_path is None:
            if os.path.exists("infinite_nature_pytorch.ckpt"):
                self.init_from_ckpt("infinite_nature_pytorch.ckpt", ignore_keys=ignore_keys)
            else:
                self.load_pretrained_weights_from_tensorflow_to_pytorch()
                torch.save(self.state_dict(), "infinite_nature_pytorch.ckpt")
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
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
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
                # print(f'Excluded weights: {name} {array.shape}')
                continue

            if 'Adam' in name:
                # print(f'Excluded weights: {name} {array.shape}')
                continue

            pointer = self
            if 'conv' in name:
                name = name.split('/')
                truncated_names = None
                for i_m, m_name in enumerate(name):
                    pointer = getattr(pointer, m_name)
                    if isinstance(pointer, Conv2D):
                        truncated_names = name[i_m+1:]
                        if truncated_names[0] == 'conv2d':
                            truncated_names = name[i_m + 2:]
                        break
                # print(truncated_names)
                if truncated_names[0] == 'kernel':
                    if len(pointer.shape) == 4:
                        # TODO: it might also be (3, 2, 0, 1), which needs double-checking
                        if len(array.shape) == 2:
                            # kernel size is 1x1
                            array = array[None, None]
                        elif len(array.shape) == 3:
                            array = array[..., None]
                        array = array.transpose(3, 2, 0, 1)
                    elif len(pointer.shape) == 2:
                        array = array.transpose(1, 0)
                    pointer.weight = torch.from_numpy(array)
                elif truncated_names[0] == 'bias':
                    if len(array.shape) == 0:
                        array = array[None,]
                    pointer.bias = torch.from_numpy(array)
                elif truncated_names[0] == 'u':
                    pointer.u = torch.from_numpy(array)
            else:
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
                            array = array.transpose(3, 2, 0, 1)
                        elif len(pointer.shape) == 2:
                            array = array.transpose(1, 0)
                    elif m_name == 'bias':
                        pointer = getattr(pointer, 'bias')
                        if len(array.shape) == 0:
                            array = array[None,]
                    else:
                        pointer = getattr(pointer, m_name)

                try:
                    if m_name != 'u':
                        assert pointer.shape == array.shape  # Catch error if the array shapes are not identical
                except AssertionError as e:
                    e.args += (pointer.shape, array.shape)
                    raise

                # print("Initialize PyTorch weight {}".format(name))
                if 'conv' not in name:
                    pointer.data = torch.from_numpy(array)
                else:
                    pointer = torch.from_numpy(array)

    def training_step(self, batch, batch_idx):
        x_src = torch.cat([batch['src_img'],
                           1/batch['src_disparity']], dim=-1).permute(0, 3, 1, 2)
        gt_tgt_rgbd = torch.cat([batch['dst_img'],
                           1/batch['dst_disparity']], dim=-1).permute(0, 3, 1, 2)
        z, mu, logvar = self.generator.style_encoding(x_src, return_mulogvar=True)
        rendered_rgbd, mask = self.render_with_projection(x_src[:, :3][:, None],
                                                          1/x_src[:, 3][:, None],
                                                          batch["Ks"][:, 0],
                                                          batch["Ks"][:, 0],
                                                          batch['T_src2tgt'])
        predicted_rgbd = self(rendered_rgbd, mask, z)
        # disc_on_generated = self.discriminate(predicted_rgbd)
        # generated_features = [f[0] for f in disc_on_generated]
        # generated_logits = [f[1] for f in disc_on_generated]
        loss_dict = compute_infinite_nature_loss(predicted_rgbd, gt_tgt_rgbd, self.discriminate, (mu, logvar), self.perceptual_loss, 'train')
        self.log_dict(loss_dict, sync_dist=True, on_step=True, on_epoch=True, rank_zero_only=True)

        opt_ae, opt_disc = self.optimizers()
        opt_ae.zero_grad()
        loss_dict['train/total_generator_loss'].backward()
        opt_ae.step()
        opt_disc.zero_grad()
        loss_dict['train/total_discriminator_loss'].backward()
        opt_disc.step()

    def validation_step(self, batch, batch_idx):
        x_src = torch.cat([batch['src_img'],
                           batch['src_disparity']], dim=-1).permute(0, 3, 1, 2)
        gt_tgt_rgbd = torch.cat([batch['dst_img'],
                           batch['dst_disparity']], dim=-1).permute(0, 3, 1, 2)

        z, mu, logvar = self.generator.style_encoding(x_src, return_mulogvar=True)
        rendered_rgbd, extrapolation_mask = self.render_with_projection(x_src[:, :3][:, None],
                                                          1/x_src[:, 3][:, None],
                                                          batch["Ks"][:, 0],
                                                          batch["Ks"][:, 0],
                                                          batch['T_src2tgt'])
        predicted_rgbd = self(rendered_rgbd, extrapolation_mask, z)
        loss_dict = compute_infinite_nature_loss(predicted_rgbd, gt_tgt_rgbd, self.discriminate, (mu, logvar), self.perceptual_loss, 'val')
        self.log_dict(loss_dict, on_epoch=True, on_step=True, sync_dist=True)

        ssim = SSIM()
        psnr = PSNR()
        ssim_score = 0
        psnr_score = 0
        ssim_visible_score = 0
        psnr_visible_score = 0

        os.makedirs(os.path.join(self.logdir, f'qualitative_res/pred'), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, f'qualitative_res/warped_inputs'), exist_ok=True)

        os.makedirs(os.path.join(self.logdir, f'qualitative_res/gt'), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, f'qualitative_res/extrapolation_mask'), exist_ok=True)
        p_loss = self.perceptual_loss(gt_tgt_rgbd[:, :3].contiguous().float(),
                                      predicted_rgbd[:, :3].contiguous().float())
        rendered_rgbd[:, :3] = torch.clip((rendered_rgbd[:, :3] + 1) / 2, 0, 1)
        predicted_rgbd[:, :3] = torch.clip((predicted_rgbd[:, :3] + 1) / 2, 0, 1)
        gt_tgt_rgbd[:, :3] = torch.clip((gt_tgt_rgbd[:, :3] + 1) / 2, 0, 1)

        for i in range(gt_tgt_rgbd.shape[0]):
            warped = torch.clip(rendered_rgbd[i, :3] * 255, 0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8).astype(
                np.float32)

            pred = torch.clip(predicted_rgbd[i, :3] * 255, 0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
                np.uint8).astype(np.float32)
            gt = torch.clip(gt_tgt_rgbd[i, :3] * 255, 0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8).astype(
                np.float32)

            res = ssim(pred, gt, (1-extrapolation_mask[i]).permute(1, 2, 0).repeat(1, 1, 3)
                       .detach().cpu().numpy() if extrapolation_mask is not None else None)
            if extrapolation_mask is None:
                curr_ssim = res
            else:
                curr_ssim, curr_visible_ssim = res
                ssim_visible_score += curr_visible_ssim

            res = psnr(pred, gt, (1-extrapolation_mask[i]).permute(1, 2, 0).repeat(1, 1, 3)
                       .detach().cpu().numpy() if extrapolation_mask is not None else None)
            if extrapolation_mask is None:
                curr_psnr = res
            else:
                curr_psnr, curr_visilble_psnr = res
                psnr_visible_score += curr_visilble_psnr

            ssim_score += curr_ssim
            psnr_score += curr_psnr

            if extrapolation_mask is not None:
                np.save(
                    os.path.join(self.logdir, f'qualitative_res/extrapolation_mask/batch_{batch_idx}_index_{i}.png'),
                    (1-extrapolation_mask[i]).permute(1, 2, 0).repeat(1, 1, 3).detach().cpu().numpy())
                PIL.Image.fromarray(warped.astype(np.uint8)).save(
                    os.path.join(self.logdir, f'qualitative_res/warped_inputs/batch_{batch_idx}_index_{i}.png'))
            PIL.Image.fromarray(pred.astype(np.uint8)).save(
                os.path.join(self.logdir, f'qualitative_res/pred/batch_{batch_idx}_index_{i}.png'))

            PIL.Image.fromarray(gt.astype(np.uint8)).save(
                os.path.join(self.logdir, f'qualitative_res/gt/batch_{batch_idx}_index_{i}.png'))
        ssim_score /= gt_tgt_rgbd.shape[0]
        psnr_score /= gt_tgt_rgbd.shape[0]
        ssim_visible_score /= gt_tgt_rgbd.shape[0]
        psnr_visible_score /= gt_tgt_rgbd.shape[0]


        rgb_l1 = F.l1_loss(predicted_rgbd[:, :3], gt_tgt_rgbd[:, :3])
        disparity_l1 = F.l1_loss(predicted_rgbd[:, 3:], gt_tgt_rgbd[:, 3:])

        self.log("val/rgb_l1", rgb_l1,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/disparity_l1", disparity_l1,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/ssim", ssim_score,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/psnr", psnr_score,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/visible_ssim", ssim_visible_score,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/visible_psnr", psnr_visible_score,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log("val/LPIPS", p_loss.mean().item(),
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def render_with_projection(self, x_src, dm_src, K_src, K_next, T_src2tgt):
        warped_depth, warped_features, extrapolation_mask = render_projection_from_srcs_fast(
            x_src,
            dm_src,
            K_next.to(self.device),
            K_src.to(self.device),
            T_src2tgt,
            src_num=x_src.shape[1])
        warped_rgbd = torch.cat([warped_features, 1/torch.clip(warped_depth, 1e-7) * (1-extrapolation_mask)], dim=1)
        return warped_rgbd, 1-extrapolation_mask

    def forward(self, rendered_rgbd, mask, encoding):
        predicted_rgbd = self.generator(rendered_rgbd, mask, encoding)
        if not self.training:
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

    def log_images(self, batch, **kwargs):
        x_src = torch.cat([batch['src_img'],
                           batch['src_disparity']], dim=-1).permute(0, 3, 1, 2)
        gt_tgt_rgbd = torch.cat([batch['dst_img'],
                                 batch['dst_disparity']], dim=-1).permute(0, 3, 1, 2)

        z, mu, logvar = self.generator.style_encoding(x_src, return_mulogvar=True)
        rendered_rgbd, extrapolation_mask = self.render_with_projection(x_src[:, :3][:, None],
                                                                        1/x_src[:, 3][:, None],
                                                                        batch["Ks"][:, 0],
                                                                        batch["Ks"][:, 0],
                                                                        batch['T_src2tgt'])
        predicted_rgbd = self(rendered_rgbd, extrapolation_mask, z)
        log = dict()
        log["warped_input"] = rendered_rgbd[:, :3]
        log["warped_disparity"] = rendered_rgbd[:, 3:]
        log["reconstructions"] = predicted_rgbd[:, :3]
        log["reconstruction_disparity"] = predicted_rgbd[:, 3:]
        log["gt_rgb"] = gt_tgt_rgbd[:, :3]
        log["gt_disparity"] = gt_tgt_rgbd[:, 3:]

        log["src_rgb"] = x_src[:, :3]
        log["src_disparity"] = x_src[:, 3:]
        return log
