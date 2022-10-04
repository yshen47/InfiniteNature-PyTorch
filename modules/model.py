from modules.generator import Generator
from modules.discriminator import PatchDiscriminator
from torch import nn


class InfiniteNature(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.generator = Generator(args)
        self.spade_discriminator_0 = PatchDiscriminator()
        self.spade_discriminator_1 = PatchDiscriminator()

    def forward(self, rendered_rgbd, mask, encoding):
        predicted_rgbd = self.generator(rendered_rgbd, mask, encoding)

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
