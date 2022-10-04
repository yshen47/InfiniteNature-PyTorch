# from interactive_demo import library
from omegaconf import OmegaConf
from data.utils.utils import instantiate_from_config
import numpy as np
import torch
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = OmegaConf.load("configs/google_earth.yaml")
    model = instantiate_from_config(config.model).eval().cuda()

    # library.download_checkpoints_and_demo_assets()
    # initial_rgbds = library.load_assets()

    # The state that we need to remember while flying:
    state = {
        'intrinsics': torch.tensor([[0.8,   0,      0.5],
                                    [0,     1.28,   0.5],
                                    [0,     0,      1]]).cuda(),
        'pose': torch.eye(4).cuda(),
        'rgbd': torch.from_numpy(np.load('rgbd.npy')).permute(2, 0, 1).cuda(),
        'start_rgbd': torch.from_numpy(np.load('start_rgbd.npy')).permute(2, 0, 1).cuda(),
        'style_noise': torch.from_numpy(np.load('style_noise.npy')).cuda(),
        'next_pose': torch.from_numpy(np.load('next_pose.npy')).cuda(),
        'direction_offset': (0, 0),  # Direction controlled by user's mouse clicks.
    }
    state['next_pose'] = torch.cat([state['next_pose'], torch.tensor([[0, 0, 0, 1]]).cuda()], 0)
    plt.imshow(state['rgbd'].permute(1, 2, 0)[:, :, :3].cpu())
    plt.show()

    plt.imshow(state['rgbd'].permute(1, 2, 0)[:, :, 3].cpu())
    plt.show()

    with torch.no_grad():
        for _ in range(10):
            x_src = state['rgbd']
            z = model.generator.style_encoding(x_src[None, ], return_mulogvar=False)
            state['T_src2tgt'] = state['next_pose'] @ torch.linalg.inv(state['pose'])
            rendered_rgbd, mask = model.render_with_projection(state['rgbd'][None, None][:, :, :3],
                                                               state['rgbd'][None, None][:, :, 3],
                                                               state['intrinsics'][None, ],
                                                               state["intrinsics"][None, ],
                                                               state['T_src2tgt'][None, ])
            predicted_rgbd = model(rendered_rgbd, mask, z)
            plt.imshow(predicted_rgbd[0].permute(1, 2, 0)[:, :, :3].cpu())
            plt.show()

            plt.imshow(predicted_rgbd[0].permute(1, 2, 0)[:, :, 3].cpu())
            plt.show()

