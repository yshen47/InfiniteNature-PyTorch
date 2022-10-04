from modules.generator import Generator
from modules.discriminator import PatchDiscriminator
import torch
import pytorch_lightning as pl


class InfiniteNature(pl.LightningModule):

    def __init__(self, generator_config, loss_config):
        super().__init__()
        self.generator = Generator(generator_config)
        self.spade_discriminator_0 = PatchDiscriminator()
        self.spade_discriminator_1 = PatchDiscriminator()

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.phase == 'codebook':
            opt_ae_parameters = list(self.encoder.parameters()) + \
                                      list(self.decoder.parameters()) + \
                                      list(self.quantize.parameters()) + \
                                      list(self.quant_conv.parameters()) + \
                                      list(self.post_quant_conv.parameters())
            if self.use_extrapolation_mask:
                opt_ae_parameters = opt_ae_parameters + list(self.conv_in.parameters())
            opt_ae = torch.optim.Adam(opt_ae_parameters,
                                      lr=lr, betas=(0.5, 0.9))
        elif self.phase == 'conditional_generation':
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) + list(self.conv_in.parameters()) if self.use_extrapolation_mask else
                                      list(self.encoder.parameters()),
                                       lr=lr, betas=(0.5, 0.9))
        else:
            raise NotImplementedError
        if self.loss.use_discriminative_loss:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
            return opt_ae, opt_disc #[opt_ae, opt_disc], []
        else:
            return opt_ae #[opt_ae, ], []

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def forward(self, rendered_rgbd, mask, encoding):
        predicted_rgbd = self.generator(rendered_rgbd, mask, encoding)
        refined_disparity = self.rescale_refined_disparity(rendered_rgbd[:, 3:], mask, predicted_rgbd[:, 3:])
        predicted_rgbd = torch.cat([predicted_rgbd[:, :3], refined_disparity], axis=1)

        disc_on_generated = self.discriminate(predicted_rgbd)
        generated_features = [f[0] for f in disc_on_generated]
        generated_logits = [f[1] for f in disc_on_generated]

        return predicted_rgbd, generated_features, generated_logits

    def discriminate(self, predicted_rgbd):
        features, logit = self.spade_discriminator_0(predicted_rgbd)

        # half size by averaging each four pixels
        x_small = (predicted_rgbd[:, :, 0::2, 0::2] + predicted_rgbd[:, :, 0::2, 1::2]
                   + predicted_rgbd[:, :, 1::2, 0::2] + predicted_rgbd[:, :, 1::2, 1::2]) / 4

        features_small, logit_small = self.spade_discriminator_1(x_small)
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
