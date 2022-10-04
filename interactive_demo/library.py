import torch

from modules.model import InfiniteNature
from configs.config import config_parser
import subprocess
import os
import pickle
import numpy as np
import tensorflow as tf
from interactive_demo import fly_camera
import IPython


def init_pretrained_model():
    args = config_parser()
    model = InfiniteNature(args)
    return load_pretrained_weights_from_tensorflow_to_pytorch(model)


def download_checkpoints_and_demo_assets():
    if not os.path.exists("ckpt"):
        subprocess.run(["wget", "https://storage.googleapis.com/gresearch/infinite_nature_public/ckpt.tar.gz"])
        subprocess.run(["tar", "-xf", "ckpt.tar.gz"])

    if not os.path.exists("autocruise_input1.pkl"):
        subprocess.run(["wget", "https://storage.googleapis.com/gresearch/infinite_nature_public/autocruise_input1.pkl"])
    if not os.path.exists("autocruise_input2.pkl"):
        subprocess.run(["wget", "https://storage.googleapis.com/gresearch/infinite_nature_public/autocruise_input2.pkl"])
    if not os.path.exists("autocruise_input3.pkl"):
        subprocess.run(["wget", "https://storage.googleapis.com/gresearch/infinite_nature_public/autocruise_input3.pkl"])


def load_assets():
    initial_rgbds = [
        pickle.load(open("autocruise_input1.pkl", "rb"))['input_rgbd'],
        pickle.load(open("autocruise_input2.pkl", "rb"))['input_rgbd'],
        pickle.load(open("autocruise_input3.pkl", "rb"))['input_rgbd']]
    return initial_rgbds


def reset(state, model:InfiniteNature, rgbd=None):
    if rgbd is None:
        rgbd = state['start_rgbd']

    height, width, _ = rgbd.shape
    aspect_ratio = width / float(height)

    rgbd = torch.from_numpy(rgbd).float().cuda().permute(2, 0, 1)
    state['rgbd'] = rgbd
    state['start_rgbd'] = rgbd
    state['pose'] = np.array(
          [[1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]],
      dtype=np.float32)
    # 0.8 focal_x corresponds to a FOV of ~64 degrees.
    state['intrinsics'] = np.array(
        [0.8, 0.8 * aspect_ratio, .5, .5],
            dtype=np.float32)
    state['direction_offset'] = (0.0, 0.0)
    state['style_noise'] = model.generator.style_encoding(rgbd[None, ])
    state['next_pose_function'] = fly_camera.fly_dynamic(state['intrinsics'], state['pose'], turn_function=(lambda _: state['direction_offset']))
    return state


def step(model, state, offsetx, offsety):
    state['direction_offset'] = (offsetx, offsety)
    next_pose = state['next_pose_function'](state['rgbd'])
    state['next_pose'] = next_pose
    z = model.generator.style_encoding(state['rgbd'][None, ])[0]
    render_rgbd, mask = render.render(
        image, pose, intrinsic, pose_next, intrinsic_next)

    state['pose'] = next_pose
    state['rgbd'] = next_rgbd
    return current_image_as_png()


def current_image_as_png():
  imgdata = tf.image.encode_png(
      tf.image.convert_image_dtype(state['rgbd'][..., :3], dtype=tf.uint8))
  return IPython.display.Image(data=imgdata.numpy())
