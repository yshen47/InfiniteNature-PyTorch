from interactive_demo import library
import torch


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = library.init_pretrained_model().cuda()
    library.download_checkpoints_and_demo_assets()
    initial_rgbds = library.load_assets()

    # The state that we need to remember while flying:
    state = {
        'intrinsics': None,
        'pose': None,
        'rgbd': None,
        'start_rgbd': initial_rgbds[0],
        'style_noise': None,
        'next_pose_function': None,
        'direction_offset': None,  # Direction controlled by user's mouse clicks.
    }

    state = library.reset(state, model)
    for _ in range(10):
        library.step(model, state, offsetx=0, offsety=0)
        pass


