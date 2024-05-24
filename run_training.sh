#!/bin/bash

# --> CONFIGURE BEFORE RUNNING JOB
NAME=$1
EPOCHS=$2
BATCH_SIZE=$3
ENC=$4
DEC=$5
DATASET=$6
TESTDIR=$7
FILTER_EXP=$8
TRAINED_PATH=$9
NGPUS=1

ENC_str=${ENC_str:1}

DEC_str=${DEC_str:1}

# --> CONFIGURE BEFORE RUNNING JOB
# #SBATCH --constraint='titan_xp|geforce_rtx_2080_ti'

# Copy code files
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
ABSPATH=/your/path/
FOLDER=./your/path/${NAME}_${timestamp}_e${EPOCHS}_bs${BATCH_SIZE}_fe${FILTER_EXP}
mkdir -p $ABSPATH/$FOLDER
mkdir -p $ABSPATH/$FOLDER/trained_model
mkdir -p $ABSPATH/$FOLDER/checkpoints

rsync -r --prune-empty-dirs --exclude ".pre-commit-config.yaml" --exclude "wandb" --exclude "outputs" --exclude "artifacts" --include="*/" --include="*.py" --include='*.yaml' --include="*.err" --include="*.out" --include="run_evaluation.sh" --include="run_training.sh" --exclude="*" "." $ABSPATH/$FOLDER

cat << EOT > "$ABSPATH/$FOLDER/train.sh"
#!/bin/bash

#SBATCH --output=$ABSPATH/$FOLDER/TRAIN-%x.%j.out
#SBATCH --error=$ABSPATH/$FOLDER/TRAIN-%x.%j.err
#SBATCH --gres=gpu:$NGPUS
#SBATCH --job-name=$NAME
#SBATCH --mail-type=BEGIN,END,FAIL

cd $ABSPATH

python -u $ABSPATH/$FOLDER/train.py $NAME $EPOCHS $BATCH_SIZE $FOLDER $ENC $DEC $DATASET $TESTDIR $FILTER_EXP $TRAINED_PATH

EOT

sbatch "$ABSPATH/$FOLDER/train.sh"
