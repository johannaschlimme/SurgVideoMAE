#!/usr/bin/env bash
set -x  # Print commands for debugging

export MASTER_PORT=$((12000 + $RANDOM % 20000))  # Set a random master port
export OMP_NUM_THREADS=1  # Control the number of threads

OUTPUT_DIR='/workspace/VideoMAEv2/output'  # Output directory for logs and checkpoints
DATA_PATH='/workspace/VideoMAEv2/data/pretrain_dataset.csv'  # Path to your dataset CSV

# GPU settings
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=4  # Reduce CPU usage for single GPU setup
PY_ARGS=${@:1}  # Capture any additional arguments passed to the script

# Run the pretraining script
python -u run_mae_pretraining.py \
    --data_path ${DATA_PATH} \
    --mask_type tube \
    --mask_ratio 0.9 \
    --decoder_mask_type run_cell \
    --decoder_mask_ratio 0.5 \
    --model pretrain_videomae_base_patch16_224 \
    --decoder_depth 4 \
    --batch_size 16 \
    --with_checkpoint \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 4 \
    --num_workers 4 \
    --opt adamw \
    --lr 6e-4 \
    --clip_grad 0.02 \
    --warmup_epochs 2 \
    --save_ckpt_freq 5 \
    --epochs 3 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    ${PY_ARGS}

# warm-up: 30
# epochs: 300
