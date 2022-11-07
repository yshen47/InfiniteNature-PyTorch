import tensorflow as tf
import torch.nn.functional


def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


def compute_infinite_nature_loss(
        generated_rgbd, gt_rgbd, discriminate_f, mu_logvar, perceptual_loss_f, split, use_discriminative_loss=True,
        last_layer=None):
    """Computes loss between a generated RGBD sequence and the ground truth.

    Lambda terms are the default values used during the original submission.

    Args:
      generated_rgbd: [B, T, H, W, 4] A batch of T-length RGBD sequences
        produced by the generator. Ranges from (0, 1)
      gt_rgbd: [B, T, H, W, 4] The ground truth sequence from a video.
        Ranges from (0, 1)
      discriminator: a discriminator function which accepts an [B, H, W, D] tensor
        and runs a discriminator on multiple scales and returns
        a list of (features, logit) for each scale.
      mu_logvar: ([B, 128], [B, 128]) A tuple of mu, log-variance features
        parameterizing the Gaussian used to sample the variational noise.

    Returns:
      A dictionary of losses. total_loss refers to the final
        loss used to optimize the generator and total_disc_loss refers to the
        loss used by the discriminator.
    """

    # discriminator returns:
    # [(scale_1_feats, scale_1_logits), (scale_2_feats, scale_2_logits), ...]
    generated_features, generated_logits = discriminate_f(generated_rgbd, use_for_discriminator_loss=True)
    real_features, real_logits = discriminate_f(gt_rgbd, use_for_discriminator_loss=True)

    disc_loss, _, _ = compute_discriminator_loss(
        real_logits, generated_logits)

    generated_features, generated_logits = discriminate_f(generated_rgbd, use_for_discriminator_loss=False)
    fool_d_loss = compute_adversarial_loss(generated_logits)

    feature_matching_loss = compute_feature_matching(real_features,
                                                     generated_features)
    kld_loss = compute_kld_loss(mu_logvar[0], mu_logvar[1])

    rgbd_loss = torch.mean(torch.abs(generated_rgbd - gt_rgbd))
    perceptual_loss = perceptual_loss_f(generated_rgbd[:, :3], gt_rgbd[:, :3]).mean()

    try:
        d_weight = calculate_adaptive_weight(rgbd_loss, fool_d_loss, last_layer=last_layer)
    except RuntimeError:
        d_weight = torch.tensor(0.0)

    loss_dict = {}
    loss_dict[f"{split}/disc_loss"] = disc_loss.detach()
    loss_dict[f"{split}/adversarial_loss"] = fool_d_loss.detach()
    loss_dict[f"{split}/feature_matching_loss"] = feature_matching_loss.detach()
    loss_dict[f"{split}/kld_loss"] = kld_loss.detach()
    loss_dict[f"{split}/perceptual_loss"] = perceptual_loss.detach()
    loss_dict[f"{split}/reconstruction_loss"] = rgbd_loss.detach()

    total_loss = (1e-2 * perceptual_loss +
                  10.0 * feature_matching_loss + 0.05 * kld_loss +
                  (1.5 if use_discriminative_loss else 0) * fool_d_loss * d_weight + 0.5 * rgbd_loss)
    total_disc_loss = 1.5 * disc_loss
    loss_dict[f"{split}/total_generator_loss"] = total_loss * d_weight
    loss_dict[f"{split}/total_discriminator_loss"] = total_disc_loss * (1 if use_discriminative_loss else 0)
    return loss_dict


def compute_kld_loss(mu, logvar):
    loss = -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))
    return loss


def compute_discriminator_loss(real_logit, fake_logit):
    """Computes the discriminator hinge loss given logits.

    Args:
      real_logit: A list of logits produced from the real image
      fake_logit: A list of logits produced from the fake image

    Returns:
      Scalars discriminator loss, adv_loss, patchwise accuracy of discriminator
      at detecting real and fake patches respectively.
    """
    # Multi-scale disc returns a list.
    real_loss, fake_loss = 0, 0
    real_total, fake_total = 0, 0
    real_correct, fake_correct = 0, 0
    for real_l, fake_l in zip(real_logit, fake_logit):
        real_loss += torch.mean(torch.nn.functional.relu(1 - real_l))
        fake_loss += torch.mean(torch.nn.functional.relu(1 + fake_l))
        real_total += torch.prod(torch.tensor(real_l.shape)).float().to(real_loss.device)
        fake_total += torch.prod(torch.tensor(fake_l.shape)).float().to(real_loss.device)
        real_correct += torch.sum(real_l >= 0)
        fake_correct += torch.sum(fake_l < 0)

    # Avg of all outputs.
    real_loss = real_loss / float(len(real_logit))
    fake_loss = fake_loss / float(len(fake_logit))
    real_accuracy = real_correct / real_total
    fake_accuracy = fake_correct / fake_total

    disc_loss = real_loss + fake_loss

    return disc_loss, real_accuracy, fake_accuracy


def compute_adversarial_loss(fake_logit):
    """Computes the adversarial hinge loss to apply to the generator.

    Args:
      fake_logit: list of tensors which correspond to discriminator logits
        on generated images

    Returns:
      A scalar of the loss.
    """
    fake_loss = 0
    for fake_l in fake_logit:
        fake_loss += -torch.mean(fake_l)

    # average of all.
    fake_loss = fake_loss / float(len(fake_logit))

    return fake_loss


def compute_feature_matching(real_feats, fake_feats):
    """Computes a feature matching loss between real and fake feature pyramids.

    Args:
      real_feats: A list of feature activations of a discriminator on real images
      fake_feats: A list of feature activations on fake images

    Returns:
      A scalar of the loss.
    """
    losses = []
    # Loop for scale
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        for i in range(len(real_feat)):
            losses.append(torch.mean(torch.abs(real_feat[i].detach() - fake_feat[i])))

    return torch.mean(torch.stack(losses))


def compute_perceptual_loss(generated, real):
    """Compute the perceptual loss between a generated and real sample.

    The input to this are RGB images ranging from [0, 255].

    build_vgg19's reference library can be found here:
    https://github.com/CQFIO/PhotographicImageSynthesis/blob/master/demo_1024p.py

    Args:
      generated: [B, H, W, 3] Generated image that we want to be perceptually
        close to real.
      real: [B, H, W, 3] Ground truth image that we want to target.

    Returns:
      A tf scalar corresponding to the perceptual loss.
    """

    # Input is [0, 255.], not necessarily clipped though
    def build_vgg19(*args):
        raise NotImplementedError

    vgg_real_fake = build_vgg19(
        tf.concat([real, generated], axis=0),
        "imagenet-vgg-verydeep-19.mat")

    def compute_l1_loss(key):
        real, fake = tf.split(vgg_real_fake[key], 2, axis=0)
        return tf.reduce_mean(tf.abs(real - fake))

    p0 = tf.zeros([])
    p1 = compute_l1_loss("conv1_2") / 2.6
    p2 = compute_l1_loss("conv2_2") / 4.8
    p3 = compute_l1_loss("conv3_2") / 3.7
    p4 = compute_l1_loss("conv4_2") / 5.6
    p5 = compute_l1_loss("conv5_2") * 10 / 1.5
    return p0 + p1 + p2 + p3 + p4 + p5
