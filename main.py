import os.path
import sys
# Make sure python can find our libraries.
sys.path.append('/home/yuan/PycharmProjects/infinite_nature/tf_mesh_renderer/mesh_renderer')
from omegaconf import OmegaConf
from data.utils.utils import instantiate_from_config
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
from modules.tensorflow_api.render import render


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = OmegaConf.load("configs/google_earth.yaml")
    model = instantiate_from_config(config.model).eval().cuda()
    # if not os.path.exists("infinite_nature_pytorch.ckpt"):
    #     torch.save(model.state_dict(), "infinite_nature_pytorch.ckpt")
    # library.download_checkpoints_and_demo_assets()
    # initial_rgbds = library.load_assets()

    # The state that we need to remember while flying:
    state = {
        'intrinsics': np.array([0.8, 1.28, 0.5, 0.5]),
        'pose': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]),
        'rgbd': torch.from_numpy(np.load('rgbd.npy')).permute(2, 0, 1).cuda(),
        'start_rgbd': torch.from_numpy(np.load('start_rgbd.npy')).permute(2, 0, 1).cuda(),
        'style_noise': torch.from_numpy(np.load('style_noise.npy')).cuda(),
        'next_pose': np.load('next_pose.npy'),
        'direction_offset': (0, 0),  # Direction controlled by user's mouse clicks.
    }
    plt.imshow(state['rgbd'].permute(1, 2, 0)[:, :, :3].cpu())
    plt.show()

    plt.imshow(state['rgbd'].permute(1, 2, 0)[:, :, 3].cpu())
    plt.show()

    with torch.no_grad():
        for _ in range(10):
            x_src = state['rgbd']
            z = model.generator.style_encoding(x_src[None, ], return_mulogvar=False)
            image = tf.constant(x_src[None,].permute(0, 2, 3, 1).detach().cpu().numpy(), dtype=tf.float32)
            pose = tf.constant(state['pose'][None,], tf.float32)
            pose_next = tf.constant(state['next_pose'][None,], tf.float32)
            intrinsic = tf.constant(state['intrinsics'][None,], tf.float32)
            intrinsic_next = tf.constant(state['intrinsics'][None,], tf.float32)
            rendered_rgbd, mask = render(image, pose, intrinsic, pose_next, intrinsic_next)
            rendered_rgbd = torch.from_numpy(rendered_rgbd.numpy()).permute(0, 3, 1, 2).cuda()
            mask = torch.from_numpy(mask.numpy()).permute(0, 3, 1, 2).cuda()
            predicted_rgbd = model(rendered_rgbd, mask, z)

            # plt.imshow(rendered_rgbd[0].permute(1, 2, 0)[:, :, :3].cpu())
            # plt.show()
            # plt.imshow(rendered_rgbd[0].permute(1, 2, 0)[:, :, 3].cpu())
            # plt.show()

            plt.imshow(predicted_rgbd[0].permute(1, 2, 0)[:, :, :3].cpu())
            plt.show()
            # plt.imshow(predicted_rgbd[0].permute(1, 2, 0)[:, :, 3].cpu())
            # plt.show()

            state['rgbd'] = predicted_rgbd[0]

