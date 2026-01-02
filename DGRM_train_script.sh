OUTDIR=logs/
DATASET=/workspace/AmbientDiffMem/AmbientDiffMemorize/datasets/cifar10/data_dimension/cifar10-32x32-64.zip

nature_noise_fixed=True
SNAP=5 # This is on ticks.
TICK=10
LOSS_TYPE=ambient_highnoise_edm_lownoise

# Relevant to me:
TRAINING_DURATION=0.5 # 0.001 == 1 kimg
SIGMA=2
LEARNING_RATE=12.5e-5

EXPNAME=cifar10-32x32-64-ddpmpp-sigma{$SIGMA}

# Resume training from checkpoint (leave empty to start from scratch)
# Specify full path to training-state-*.pt file
# Example: RESUME=/workspace/AmbientDiffMem/AmbientDiffMemorize/logs/00001-.../training-state-000200.pt
#RESUME="/workspace/AmbientDiffMem/AmbientDiffMemorize/logs/00010-cifar10-32x32-1000-uncond-ncsnpp-edm-gpus1-batch128-fp32-V2cvj/training-state-003000.pt"

NPROC_PER_NODE=1

# Clean up Python cache files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

# Add resume flag if specified
RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    if [ ! -f "$RESUME" ]; then
        echo "Error: Resume checkpoint not found: $RESUME"
        exit 1
    fi
    RESUME_FLAG="--resume=$RESUME"
    echo "Resuming from checkpoint: $RESUME"
fi

# Launch distributed training with specified number of GPUs
# Directory to save model checkpoints and logs
# Path to training dataset
# Unconditional generation (no class labels)
# Neural network architecture
# Total batch size across all GPUs
# Total training duration
# Channel multipliers for each resolution
# Dropout probability
# Data augmentation probability (set to 0 to disable)
# Learning rate
# Nature noise level sigma
# Whether to use nature noise fixed for each example
# Checkpoint interval
# Type of training loss (Options: edm (baseline), ambient_highnoise_edm_lownoise)
# Experiment ID
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE DGRM_train.py \
    --outdir=$OUTDIR \
    --data=$DATASET \
    --arch=ddpmpp \
    --batch=32 \
    --duration=$TRAINING_DURATION \
    --sigma=$SIGMA \
    --nature_noise_fixed=$nature_noise_fixed \
    --tick=$TICK \
    --snap=$SNAP \
    --loss_type=$LOSS_TYPE \
    --augment=0 \
    --expr_id={$EXPNAME}_ffhq_cp1_sigma{$SIGMA} \
    $RESUME_FLAG