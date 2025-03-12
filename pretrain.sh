#!/bin/bash -eux

#SBATCH --job-name=videomae_pt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=johanna.schlimme@student.hpi.de
#SBATCH --partition=gpupro
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --output=output/videomae_pt_%j.log

# Print start time and node info
date
hostname

# Manually set CUDA paths
export CUDA_VERSION="11.5"
export PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH

# Disable DDP Optimizer for single GPU
export TORCHDYNAMO_DISABLE_DDP_OPTIMIZER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Define user and environment variables
user="johanna.schlimme"
condaenv="videomae"
project_root="/dhc/home/$user/videomae"
script_name="scripts/pretrain/ophnet_pt.sh"

# Move to project root
cd $project_root

# Activate Conda environment
eval "$(conda shell.bash hook)"
source /dhc/home/$user/conda3/bin/activate
conda activate $condaenv

# Ensure the script has execution permissions
chmod +x $script_name

# Run training script
bash $script_name --wandb

# Print end time
date
