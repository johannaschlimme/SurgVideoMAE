#!/usr/bin/env bash
set -x  # Print commands for debugging
set -e  # Exit on error

# Set default values if not passed via env
OUTPUT_DIR="${OUTPUT_DIR:-output}"
DATA_PATH="${DATA_PATH:-data/pretrain_dataset.csv}"
GPUS="${GPUS:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PY_ARGS="${@}"                   # All extra arguments passed to this script

# Launch training with DDP
torchrun --nnodes=1 --nproc_per_node=${GPUS} run_mae_pretraining.py \
    --data_path ${DATA_PATH} \
    --mask_type tube \
    --mask_ratio 0.75 \
    --decoder_mask_type run_cell \
    --decoder_mask_ratio 0.5 \
    --model pretrain_videomae_small_patch16_224 \
    --decoder_depth 4 \
    --batch_size 32 \
    --with_checkpoint \
    --num_frames 8 \
    --sampling_rate 4 \
    --num_sample 4 \
    --num_workers ${CPUS_PER_TASK} \
    --opt adamw \
    --lr 6e-4 \
    --min_lr 5e-5 \
    --clip_grad 0.02 \
    --warmup_epochs 15 \
    --save_ckpt_freq 10 \
    --epochs 150 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    ${PY_ARGS}
