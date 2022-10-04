import torch
from modules.model import InfiniteNature
import subprocess
import os
import pickle
import numpy as np
import tensorflow as tf
from interactive_demo import fly_camera


def create_vertices_intrinsics(disparity, intrinsics):
    """3D mesh vertices from a given disparity and intrinsics.

    Args:
       disparity: [B, H, W] inverse depth
       intrinsics: [B, 4] reference intrinsics

    Returns:
       [B, L, H*W, 3] vertex coordinates.
    """
    # Focal lengths
    fx = intrinsics[:, 0]
    fy = intrinsics[:, 1]
    fx = fx[Ellipsis, tf.newaxis, tf.newaxis]
    fy = fy[Ellipsis, tf.newaxis, tf.newaxis]

    # Centers
    cx = intrinsics[:, 2]
    cy = intrinsics[:, 3]
    cx = cx[Ellipsis, tf.newaxis]
    cy = cy[Ellipsis, tf.newaxis]

    batch_size, height, width = disparity.shape.as_list()
    vertex_count = height * width

    i, j = tf.meshgrid(tf.range(width), tf.range(height))
    i = tf.cast(i, tf.float32)
    j = tf.cast(j, tf.float32)
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
    # 0.5 is added to get the position of the pixel centers.
    i = (i + 0.5) / width
    j = (j + 0.5) / height
    i = i[tf.newaxis]
    j = j[tf.newaxis]

    depths = 1.0 / tf.clip_by_value(disparity, 0.01, 1.0)
    mx = depths / fx
    my = depths / fy
    px = (i - cx) * mx
    py = (j - cy) * my

    vertices = tf.stack([px, py, depths], axis=-1)
    vertices = tf.reshape(vertices, (batch_size, vertex_count, 3))
    return vertices


def download_checkpoints_and_demo_assets():
    if not os.path.exists("ckpt"):
        subprocess.run(["wget", "https://storage.googleapis.com/gresearch/infinite_nature_public/ckpt.tar.gz"])
        subprocess.run(["tar", "-xf", "ckpt.tar.gz"])

    if not os.path.exists("autocruise_input1.pkl"):
        subprocess.run(
            ["wget", "https://storage.googleapis.com/gresearch/infinite_nature_public/autocruise_input1.pkl"])
    if not os.path.exists("autocruise_input2.pkl"):
        subprocess.run(
            ["wget", "https://storage.googleapis.com/gresearch/infinite_nature_public/autocruise_input2.pkl"])
    if not os.path.exists("autocruise_input3.pkl"):
        subprocess.run(
            ["wget", "https://storage.googleapis.com/gresearch/infinite_nature_public/autocruise_input3.pkl"])


def load_assets():
    initial_rgbds = [
        pickle.load(open("autocruise_input1.pkl", "rb"))['input_rgbd'],
        pickle.load(open("autocruise_input2.pkl", "rb"))['input_rgbd'],
        pickle.load(open("autocruise_input3.pkl", "rb"))['input_rgbd']]
    return initial_rgbds


def reset(state, model: InfiniteNature, rgbd=None):
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
    state['style_noise'] = model.generator.style_encoding(rgbd[None,])
    state['next_pose_function'] = fly_camera.fly_dynamic(state['intrinsics'], state['pose'],
                                                         turn_function=(lambda _: state['direction_offset']))
    return state
