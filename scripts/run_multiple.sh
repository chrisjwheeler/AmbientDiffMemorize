#!/usr/bin/env bash
set -eu

DATASETS_DIR=/workspace/AmbientDiffMem/AmbientDiffMemorize/datasets/cifar10/data_dimension
RESULTS_DIR=logs_exp2/

# --------- Hyperparams (tweakable) ----------
SIGMAS=("0.4")   # The noise levels to run the experiment at.
SIZES=("256")    # The dataset sizes to run the experiment at.
TICK=10
SNAP=10


# --------- Fixed params  ----------
LR=0.00000125
EPOCH_CONST=8000
BATCH=32
ARCH="ddpmpp"

# Duration is in MIMG (millions of images)
#  - 0.001 MIMG = 1 kimg = 1,000 images seen (good for smoke)
#  - 0.01  MIMG = 10 kimg = 10,000 images seen (short, visible)


mkdir -p "$RESULTS_DIR"


# Loop through sizes then noises.

for N in "${SIZES[@]}"; do
  echo "Running for N=${N}"
  # Check if the dataset path exists.
  DATA_PATH="${DATASETS_DIR}/cifar10-32x32-${N}.zip"
  if [ ! -f "$DATA_PATH" ]; then
    echo "Dataset path $DATA_PATH not found; skipping"
    continue
  fi

  for sigma in "${SIGMAS[@]}"; do
    echo "Running for sigma=${sigma}"
    RUNNAME="N${N}_sigma${sigma}"
    OUTDIR="${RESULTS_DIR}/${RUNNAME}"
    
    # Calculate the duration and learning rate.
    DURATION_MIMG=$(awk -v epoch=$EPOCH_CONST -v n=$N "BEGIN {printf \"%.6f\", (epoch * n) / 1000000}")

    
    echo "LR=${LR}"
    echo "DURATION_MIMG=${DURATION_MIMG}"
    
    mkdir -p "$OUTDIR"
    echo "Running: sigma=${sigma} N=${N} -> ${OUTDIR}"

    torchrun --standalone --nproc_per_node=1 DGRM_train.py \
    --outdir=$OUTDIR \
    --data=$DATA_PATH \
    --arch=$ARCH \
    --batch=$BATCH \
    --duration=$DURATION_MIMG \
    --sigma=$sigma \
    --lr=$LR \
    --nature_noise_fixed=True \
    --tick=$TICK \
    --snap=$SNAP

  done
done

