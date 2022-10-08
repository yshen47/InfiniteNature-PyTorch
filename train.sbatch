#!/bin/sh
#
#SBATCH --time=3-0:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=shenlong
#SBATCH --job-name=in_clevr_neurips
#SBATCH --mem 10000
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/in_clevr_neurips_1e-3.%j.out
#SBATCH --error=logs/in_clevr_neurips_1e-3.%j.err
#SBATCH --mail-user=yshen47@illinois.edu
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
source /home/yshen47/.bashrc
conda activate geofree
#python train_generative_sensing_model.py --base configs/codebooks/google_earth_vqgan.yaml --gpus 0,1,2,3 -t True --resume logs/2022-10-02T18-22-39_google_earth_vqgan/checkpoints/last.ckpt
python train.py --config_path=configs/clevr-infinite.yaml
