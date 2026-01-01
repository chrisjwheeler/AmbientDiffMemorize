OUTDIR=logs/
DATASET=ffhq-64x64-300.zip
EXPNAME=test_upload_code

nature_noise_fixed=True
LOSS_TYPE=ambient_highnoise_edm_lownoise

# Relevant to me:
TRAINING_DURATION=30
SIGMA=0.5
TICK=100
NPROC_PER_NODE=1

# Unsure how to scale:
# CORRUPT_PROB must be set to 1, for their implementation to work.
DP=0.1

# Clean up Python cache files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

# Launch distributed training with specified number of GPUs
# Directory to save model checkpoints and logs
# Path to training dataset
# Unconditional generation (no class labels)
# Neural network architecture
# Total batch size across all GPUs
# Total training duration
# Channel multipliers for each resolution
# Dropout probability
# Data augmentation probability  
# Learning rate
# Nature noise level sigma
# Whether to use nature noise fixed for each example
# Checkpoint interval
# Type of training loss (Options: edm (baseline), ambient_highnoise_edm_lownoise)
# Experiment ID
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE DGRM_train.py \
    --outdir=$OUTDIR \
    --data=$DATASET \
    --cond=0 \
    --arch=ncsnpp \
    --batch=256 \
    --duration=$TRAINING_DURATION \
    --cres=1,2,2,2 \
    --dropout=0.05 \
    --augment=0.15 \
    --lr=1e-4 \
    --sigma=$SIGMA \
    --nature_noise_fixed=$nature_noise_fixed \
    --tick=$TICK \
    --loss_type=$LOSS_TYPE \
    --expr_id={$EXPNAME}_ffhq_cp1_sigma{$SIGMA}