#!/bin/bash
#SBATCH --job-name=videomae_ddp_test             # Job name
#SBATCH --partition=lrz-dgx-a100-80x8            # Partition for A100 GPUs
#SBATCH --gres=gpu:4                             # Request 4 GPUs
#SBATCH --cpus-per-task=8                        # 8 CPU cores per task (2 per GPU)
#SBATCH --mem=128G                               # Request 128 GB of RAM
#SBATCH --time=04:00:00                          # Max runtime of 4 hours
#SBATCH --output=output/videomae_%j.out          # Output log file (%j = Job ID)
#SBATCH --error=output/videomae_%j.err           # Error log file
#SBATCH --ntasks=4                               # 4 tasks (1 per GPU)
#SBATCH --nodes=1                                # All tasks on a single node

# Debug info
set -x

# WandB options
export WANDB_MODE=online                         # Online logging
export WANDB_CACHE_DIR=/tmp                      # Helps with temporary data caching

# Start the Enroot container with the mounted directory
enroot start --rw -m /dss/dsshome1/0E/ra48zaq2/videomae:/workspace/videomae videomae << 'EOF'
    # Activate the Conda environment within the container
    source /opt/conda/etc/profile.d/conda.sh
    conda activate videomae

    # Navigate to the project directory
    cd /workspace/videomae

    # Launch distributed training
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port=$((12000 + RANDOM % 20000)) \
        run_mae_pretraining.py \
        --data_path /workspace/videomae/data/pretrain_dataset.csv \
        --mask_type tube \
        --mask_ratio 0.9 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 16 \                             # Per GPU batch size
        --with_checkpoint \
        --num_frames 32 \
        --sampling_rate 4 \
        --num_sample 4 \
        --num_workers 4 \
        --opt adamw \
        --lr 6e-4 \
        --clip_grad 0.02 \
        --warmup_epochs 3 \
        --save_ckpt_freq 5 \
        --epochs 5 \
        --log_dir /workspace/videomae/output \
        --output_dir /workspace/videomae/output \
        --wandb \
        --wandb_project "VideoMAEv2" \
        --wandb_name "pretrain_run_A100_ddp"
EOF
