#!/bin/bash
# Submit a training job to SLURM.
# Usage: ./scripts/run_training.sh MODEL EPOCHS BATCH_SIZE ENC_BLOCKS DEC_BLOCKS DATASET TEST_DIR FILTER_EXP [CHECKPOINT]
# Example: ./scripts/run_training.sh SplitterNet 20 16 1,1,1,1 1,1,1,1 /data/patches /data/test_set 5
set -euo pipefail

NAME=$1
EPOCHS=$2
BATCH_SIZE=$3
ENC=$4
DEC=$5
DATASET=$6
TESTDIR=$7
FILTER_EXP=$8
CHECKPOINT=${9:-}
NGPUS=1

# --> CONFIGURE BEFORE RUNNING JOB
ABSPATH=/your/path
# <--

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
FOLDER=$ABSPATH/runs/${NAME}_${timestamp}_e${EPOCHS}_bs${BATCH_SIZE}_fe${FILTER_EXP}
mkdir -p "$FOLDER"

# Snapshot the code so later edits do not affect queued jobs.
rsync -r --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" "./" "$FOLDER/code"

CHECKPOINT_ARG=""
if [ -n "$CHECKPOINT" ] && [ "$CHECKPOINT" != "None" ]; then
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
fi

cat << EOT > "$FOLDER/train.sh"
#!/bin/bash
#SBATCH --output=$FOLDER/TRAIN-%x.%j.out
#SBATCH --error=$FOLDER/TRAIN-%x.%j.err
#SBATCH --gres=gpu:$NGPUS
#SBATCH --job-name=$NAME
#SBATCH --mail-type=BEGIN,END,FAIL

python -u $FOLDER/code/train.py --model $NAME --epochs $EPOCHS --batch-size $BATCH_SIZE \\
    --enc-blocks $ENC --dec-blocks $DEC --dataset $DATASET --test-dir $TESTDIR \\
    --filter-exp $FILTER_EXP --output-dir $FOLDER $CHECKPOINT_ARG
EOT

sbatch "$FOLDER/train.sh"
