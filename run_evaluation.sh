#!/bin/bash

# --> CONFIGURE BEFORE RUNNING JOB
TRAINED_PATH=$1
MODE=$2
TEST_DATA=$3
NGPUS=1

# --> CONFIGURE BEFORE RUNNING JOB

# Copy code files
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
ABSPATH=/home/rflepp/
FOLDER=./scratch_second/outputs/evaluate_${timestamp}
mkdir -p $ABSPATH/$FOLDER

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/itet-stor/rflepp/net_scratch/conda/lib

rsync -r --prune-empty-dirs --exclude ".pre-commit-config.yaml" --exclude "wandb" --exclude "outputs" --exclude "artifacts" --include="*/" --include="*.py" --include='*.yaml' --include="*.err" --include="*.out" --include="run_evaluation.sh" --include="run_training.sh" --exclude="*" "." $ABSPATH/$FOLDER

cat << EOT > "$ABSPATH/$FOLDER/train.sh"
#!/bin/bash

#SBATCH --output=$ABSPATH/$FOLDER/TRAIN-%x.%j.out
#SBATCH --error=$ABSPATH/$FOLDER/TRAIN-%x.%j.err
#SBATCH --gres=gpu:$NGPUS
#SBATCH --mem-per-gpu=20g
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x|geforce_rtx_2080_ti|geforce_gtx_1080_ti|titan_x'
#SBATCH --job-name=$NAME
#SBATCH --mail-type=BEGIN,END,FAIL

cd $ABSPATH

python -u $ABSPATH/$FOLDER/evaluate.py $TRAINED_PATH $MODE $TEST_DATA

EOT

sbatch "$ABSPATH/$FOLDER/train.sh"

