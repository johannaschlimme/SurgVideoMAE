#!/usr/bin/env bash
set -x  # Print commands for debugging

# Define environment variables (can be overridden by user)
OUTPUT_DIR="${OUTPUT_DIR:-output}"  # Default: "output"
DATA_PATH="${DATA_PATH:-data/pretrain_dataset.csv}"  # Default: "data/pretrain_dataset.csv"

# GPU and CPU settings
GPUS="${GPUS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"  # Reduce CPU usage for single GPU setup
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
    --warmup_epochs 30 \
    --save_ckpt_freq 25 \
    --epochs 300 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    ${PY_ARGS}