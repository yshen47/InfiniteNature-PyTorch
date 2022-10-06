# InfiniteNature-PyTorch

Unofficial PyTorch implementation of this ICCV 2021 Oral paper: [Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image](https://arxiv.org/abs/2012.09855) Check their project page for details: https://infinite-nature.github.io/

## Pre-trained Checkpoints
Download the converted Pytorch weight from their tensorflow checkpoint from [this link](https://drive.google.com/file/d/14y4OKighwK82YpMxt6H4bPr_kNy2AB_N/view?usp=sharing) and save in project root directory.

## Environment
This codebase runs successfully in Python 3.7.13 and Ubuntu 22.03. First create your conda environment and then run
```angular2html
pip install -r requirement.txt
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==1.7.7
```

BUILDING TF Mesh Renderer (Copy from their original codebase Readme.md)

We use the differentiable renderer from [here] (https://github.com/google/tf_mesh_renderer). We use gcc to build the library instead of their Bazel instructions to make it compatible with Tensorflow 2.0. To download and build:
source download_tf_mesh_renderer.sh

tf_mesh_renderer was originally built for Tensorflow < 2.0.0, however we have prepared a small patch which upgrades the functions we use to work in Tensorflow 2.2.0. This means that the other parts of tf_mesh_renderer are still version incompatible.

And make sure you add the following scripts at the top of your main script, e.g., modify line 4 of main.py to direct to tf_mesh_renderer

```
# Make sure dynamic linking can find tensorflow libraries.
# os.system('ldconfig ' + tf.sysconfig.get_lib()) # You can also do in the terminal, cuz it might require root access. e.g. sudo ldconfig xxx, xxx can be printed out in python console from tf.sysconfig.get_lib()

# Make sure python can find our libraries. It needs absolute path, relative path doesn't work for me
sys.path.append('/.../infinite_nature') 
sys.path.append('/.../infinite_nature/tf_mesh_renderer/mesh_renderer')

# Make sure the mesh renderer library knows where to load its .so file from.
os.environ['TEST_SRCDIR'] = '/.../infinite_nature'
```
## Inference

To run their similar Colab unrolling but using Pytorch, you can run the following command with fixed pose and next pose. 
```angular2html
    python main.py
```

Feel free to modify the next pose to have more interesting trajectory. We manually check our output image for one-step extrapolation is exactly the same as those in Tensorflow

## Train on your own dataset
For each dataset, we need to first prepare a config file and then write a customized data loader. Make sure you follow our provided config and dataloader example for GoogleEarth-Infinite dataset. The training is wrapped in Pytorch-lightning framework

```angular2html 
    python train.py
```

For details about GoogleEarth-Infinite dataset, please check the paper [SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping](https://github.com/yshen47/SGAM).

