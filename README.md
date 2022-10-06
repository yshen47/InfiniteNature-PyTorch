# InfiniteNature-PyTorch

Unofficial PyTorch implementation of the paper: [Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image](https://arxiv.org/abs/2012.09855)

Check their project page for details: https://infinite-nature.github.io/
Downloaded the converted Pytorch weight from their tensorflow checkpoint from [this link](https://drive.google.com/file/d/14y4OKighwK82YpMxt6H4bPr_kNy2AB_N/view?usp=sharing) and save in project root directory.

## Environment
This codebase runs successfully in Python 3.7.13 and Ubuntu 22.03. First create your conda environment and then run
```angular2html
pip install -r requirement.txt
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==1.7.7
```
## Inference

To run their similar Colab unrolling but using Pytorch, you can run the following command with fixed pose and next pose. 
```angular2html
    python main.py
```

Feel free to modify the next pose to have more interesting trajectory. We manually check our output image for one-step extrapolation is exactly the same as those in Tensorflow

## Train
For each dataset, you need to first prepare a config file and write your own data loader. Make sure you follow our provided config and dataloader example on google_earth dataset. 
The training is wrapped in Pytorch-lightning framework

```angular2html 
    python train.py
```
