#!/bin/bash
# Submit an evaluation job to SLURM.
# Usage: ./scripts/run_evaluation.sh MODEL_PATH TEST_DIR [MODEL_NAME]
# MODEL_NAME is required when MODEL_PATH is a .h5 weights file, e.g. SplitterNet.
set -euo pipefail

MODEL_PATH=$1
TEST_DIR=$2
MODEL_NAME=${3:-}
NGPUS=1

# --> CONFIGURE BEFORE RUNNING JOB
ABSPATH=/your/path
# <--

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
FOLDER=$ABSPATH/runs/evaluate_${timestamp}
mkdir -p "$FOLDER"

rsync -r --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" "./" "$FOLDER/code"

MODEL_ARG=""
if [ -n "$MODEL_NAME" ]; then
    MODEL_ARG="--model $MODEL_NAME"
fi

cat << EOT > "$FOLDER/evaluate.sh"
#!/bin/bash
#SBATCH --output=$FOLDER/EVAL-%x.%j.out
#SBATCH --error=$FOLDER/EVAL-%x.%j.err
#SBATCH --gres=gpu:$NGPUS
#SBATCH --job-name=evaluate
#SBATCH --mail-type=BEGIN,END,FAIL

python -u $FOLDER/code/evaluate.py $MODEL_PATH $TEST_DIR $MODEL_ARG
EOT

sbatch "$FOLDER/evaluate.sh"
