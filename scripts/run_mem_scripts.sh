#!/usr/bin/env bash
set -eu

DATASETS_DIR=/workspace/AmbientDiffMem/AmbientDiffMemorize/datasets/cifar10/data_dimension
RESULTS_DIR=/workspace/AmbientDiffMem/AmbientDiffMemorize/logs_exp2

# --------- Hyperparams (tweakable) ----------
SIGMAS=("0" "0.4" "2")   # The noise levels to run the experiment at.
SIZES=("128" "256" "1000")    # The dataset sizes to run the experiment at.


# Duration is in MIMG (millions of images)
#  - 0.001 MIMG = 1 kimg = 1,000 images seen (good for smoke)
#  - 0.01  MIMG = 10 kimg = 10,000 images seen (short, visible)


mkdir -p "$RESULTS_DIR"


# Loop through sizes then noises.

for N in "${SIZES[@]}"; do
  for sigma in "${SIGMAS[@]}"; do
    echo "Searching for sigma=${sigma} and N=${N}"

    CHECKPOINT_PATH="${RESULTS_DIR}/N${N}_sigma${sigma}"
    if [ ! -d "$CHECKPOINT_PATH" ]; then
      echo "Results path $CHECKPOINT_PATH not found; skipping"
      continue
    fi

    DATA_PATH="${DATASETS_DIR}/cifar10-32x32-${N}.zip"
    if [ ! -f "$DATA_PATH" ]; then
      echo "Dataset path $DATA_PATH not found; skipping"
      continue
    fi

    torchrun --standalone --nproc_per_node=1 -- mem_ratio.py --expdir=$CHECKPOINT_PATH --knn-ref=$DATA_PATH --log=memtest/mem_traj.log --seeds=0-999 --subdirs --batch=128

  done
done

