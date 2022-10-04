from generator import Generator
import configargparse
import torch
from network_utils import load_pretrained_weights_from_tensorflow_to_pytorch
from discriminator import MultiscaleDiscriminator

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)

    parser.add_argument("--experiments-root", type=str, default="./experiments")
    parser.add_argument("--experiment-name", type=str, default="train_ours_without_gt_depth")
    parser.add_argument("--mode", type=str, default='joint_inpainting',
                        help='[depth_inpainting, texture_inpainting, joint_inpainting, aggregation_net]')

    # common ml param
    parser.add_argument("--epoch", type=int, default=30000, help="")
    parser.add_argument("--batch_size", type=int, default=2, help="") # currently, only 1 is possible
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=[0])
    parser.add_argument("--learning_rate", type=float, default=5e-4) # DONE: when batch size = 16, and lr is 0.01, it gives empty image

    # data
    parser.add_argument("--datadir", type=str, default='dataset/tat',
                        help='input data directory')
    parser.add_argument("--use_psrc_count", type=bool, default=False,
                        help='input data directory')
    parser.add_argument("--data_division", type=int, default=5)
    parser.add_argument("--cropped_height", type=int, default=256, help="size of cropped_height image height")
    parser.add_argument("--cropped_width", type=int, default=480, help="size of cropped_width image width")
    parser.add_argument("--psrc_count_downsample_scale", type=int, default=3, help="psrc_count_downsample_scale")
    parser.add_argument("--train_set", type=list, default=['training/Truck'])
    parser.add_argument("--val_set", type=list, default=["training/Truck"])
    parser.add_argument("--test_set", type=list, default=["training/Truck"])

    parser.add_argument("--use_aggregation_net", type=bool, default=False, # Shenlong said to first turn this off
                        help='Assume use_tgt_gt_feature is False. If true, then use aggregation net to transform projected source features. otherwise, use projected features from source views')
    parser.add_argument("--use_geometry", type=bool, default=True,
                        help='If true, then use depth inpainter. otherwise, not including depth as input to texture generator')
    parser.add_argument("--use_gt_depth", type=bool, default=True,
                        help='')
    parser.add_argument("--use_rgb_feature", type=bool, default=False,
                        help='')
    parser.add_argument("--use_logvar", type=bool, default=False,
                        help='')
    parser.add_argument("--trainable_parameters", type=list, default=["encoder", "depth_inpainter", "texture_inpainter"],
                        help='subset of [encoder, depth_inpainter, texture_inpainter, psrc]')
    parser.add_argument("--dynamic_model_type", type=str, default='spade',
                        help='[spade, residual_unet]')

    parser.add_argument('--num_channel', type=int, default=32,
                        help='Generator num_channel')

    parser.add_argument('--num_D', type=int, default=2,
                        help='number of discriminators to be used in multiscale')
    parser.add_argument("--discriminator_update_freq", type=int, default=50,
                        help='[discriminator_update_freq]')
    parser.add_argument("--src-num", type=int, default=3,
                        help='number of source images.')
    parser.add_argument("--trajectory_length", type=int, default=7,
                        help='sample trajectory length')
    parser.add_argument("--one_step_extrapolation_max_distance", type=int, default=1.2,
                        help='use to build graph')
    parser.add_argument("--embedding_size", type=str, default=256,
                        help='encoder embedding size')
    parser.add_argument("--num_hidden", type=str, default=128,
                        help='hidden dimension size in SPADE module')
    parser.add_argument("--use_spectral_norm", type=bool, default=False,
                        help='')
    parser.add_argument('--no_ganFeat_loss', action='store_true',
                        help='if specified, do *not* use discriminator feature matching loss')

    # parser.add_argument("--lambda_pixel_disparity", type=float, default=1, help="Loss weight of L1 pixel-wise rgb loss between translated image and real image")
    parser.add_argument("--lambda_pixel_rgb", type=float, default=1, help="Loss weight of L1 pixel-wise rgb loss between translated image and real image")
    parser.add_argument("--lambda_pixel_depth", type=float, default=1, help="Loss weight of L1 pixel-wise depth loss between translated image and real image")
    parser.add_argument("--lambda_aggregation_net", type=float, default=0, help="Loss weight of aggregation net loss")
    parser.add_argument("--lambda_loss_multi_source_consistency_loss", type=float, default=0.01, help="Loss weight of consistency loss at psrc poses")
    parser.add_argument("--lambda_patch_gan_D", type=float, default=1, help="Loss weight of patch gan loss between translated image and real image")
    parser.add_argument("--lambda_mask_gan_D", type=float, default=1, help="Loss weight of patch gan loss between translated image and real image")
    parser.add_argument("--lambda_patch_gan_G", type=float, default=1)
    parser.add_argument("--lambda_mask_gan_G", type=float, default=1)
    parser.add_argument("--lambda_loss_patch_gan_psrc_G", type=float, default=1)

    parser.add_argument("--lambda_loss_kld", type=float, default=0.05)
    parser.add_argument("--lambda_loss_vgg", type=float, default=10)
    parser.add_argument('--lambda_feat', type=float, default=10, help='weight for feature matching loss')

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = config_parser()
    g = Generator(args)
    d = MultiscaleDiscriminator()
    g = load_pretrained_weights_from_tensorflow_to_pytorch(g)
    rendered_rgbd = torch.rand([2, 4, 160, 256])
    encoding = torch.rand([2, 4, 160, 256])
    mask = torch.rand([2, 1, 160, 256])
    predicted_rgbd = g(rendered_rgbd, mask, encoding)
    print(predicted_rgbd.shape)

    disc_on_generated = d(predicted_rgbd)
    generated_features = [f[0] for f in disc_on_generated]
    generated_logits = [f[1] for f in disc_on_generated]

    print("Finished!")


