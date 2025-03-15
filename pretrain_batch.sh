#!/bin/bash -eux

#SBATCH --job-name=videomae_pretrain
#SBATCH --mail-user=johanna.schlimme@campus.lmu.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=output/videomae_%j.log

# Print start time and node info
date
hostname

# Define paths
USER_ID="ra48zaq2"
PROJECT_ROOT="/dss/dsshome1/0E/$USER_ID/videomae"
CONTAINER_IMAGE="/dss/dsshome1/0E/$USER_ID/videomae.sqsh"
CONTAINER_NAME="videomae"

# Check if the Enroot container already exists
if enroot list | grep -q "^$CONTAINER_NAME$"; then
    echo "Enroot container '$CONTAINER_NAME' already exists. Skipping creation."
else
    echo "Creating Enroot container from image..."
    enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
fi

echo "Starting Enroot container..."
enroot start --rw -m $PROJECT_ROOT:/workspace/videomae $CONTAINER_NAME << 'EOF'
    # Load Conda environment inside container
    eval "$(conda shell.bash hook)"
    conda activate videomae

    # Ensure all required packages are installed
    conda install --yes wandb
    pip install wandb  # Ensures latest version

    # Move to workspace
    cd /workspace/videomae

    # Ensure script is executable
    chmod +x scripts/pretrain/ophnet_pt.sh

    # Run training script
    bash scripts/pretrain/ophnet_pt.sh --wandb

    # Print completion time
    date
