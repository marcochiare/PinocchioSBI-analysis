#!/bin/bash
#SBATCH --account=mls2026
#SBATCH --partition=a100-gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=SBI_training
#SBATCH --output=../logs/training_sbi/slurm-%x_%j.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# ====================== #
# MODULES & LIBRARIES
# ====================== #

#module load python/3.10.8--gcc--8.5.0

source $WORK/Code/envs/sbi/bin/activate
#source $HOME/dott/envs/sbi/bin/activate
#source $HOME/MLS2026/mchiaren/sbi/bin/activate

#set -euo pipefail

# ====================== #
# PATHS & NAMES
# ====================== #

# Where to save the model weights 
WEIGHTS_DIR="${WEIGHTS_DIR:-../weights}"

# Directory containing all the summary statistics
SUMSTATS_DIR="${SUMSTATS_DIR:-$WORK/stats}"

# Cosmo Param file
COSMO_FILE="${COSMO_FILE:-../SobolSeq/models_parameters_3dim.txt}"

# Python script
PY_SCRIPT="${PY_SCRIPT:-../scripts/train.py}"

# ====================== #
# USER PARAMETERS
#
# Edit here 
# ====================== #

# Parameter list to use (from COSMO_FILE)
PARAMS=(
	"Om0"
	"sigma8"
	"h"
)

DATATYPE="${DATATYPE:-counts_lmb_z}"
EXTRA_TAG="DR1-sr-ext"
VAL_FRAC=0.05
TEST_FRAC=0.1
PCA=0
ENSEMBLE=1
NUM_ENSEMBLE=5
EMBEDDING_NET=1
WHICH_EMBED="FC"
HALVE_FC=1
NORMALISE=0

# ====================== #
# ARGS (edit)
# ====================== #

ARGS=(
	--parameters-path "${COSMO_FILE}"
	--sumstats-path "${SUMSTATS_DIR}"
	--weights-path "${WEIGHTS_DIR}"
	--extra-tag "${EXTRA_TAG}"
	--probe "${DATATYPE}"
	--val-frac "${VAL_FRAC}"
	--test-frac "${TEST_FRAC}"
	--which-param "${PARAMS[@]}"
	--which-embedding "${WHICH_EMBED}"
	--num-ensemble "${NUM_ENSEMBLE}"
	--density-est maf
	--num-transforms 7
	--hidden-features 256
	--cnn-hidden-features 128
	--fc-hidden-features 64
	--fc-out-dimension 8
	--fc-layers 1
	--batchsize 64
	--learning-rate 5.e-04
)

# do not edit here
if [ "$PCA" -eq 1 ]; then
	ARGS+=(--do-pca)
fi

if [ "$ENSEMBLE" -eq 1 ]; then
	ARGS+=(--ensemble-training)
fi

if [ "$EMBEDDING_NET" -eq 1 ]; then
	ARGS+=(--use-embedding)
fi

if [ "$NORMALISE" -eq 1 ]; then
	ARGS+=(--normalise)
fi

if [ "$HALVE_FC" -eq 1 ]; then
	ARGS+=(--fc-halve)
fi

# ====================== #
# PRINTS
# ====================== #

echo -e "\033[32m[JOB]\033[0m ${SLURM_JOB_ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m SUMSTATS_DIR ....... = ${SUMSTATS_DIR}" 
echo -e "\033[32m[JOB]\033[0m COSMO_FILE ......... = $(realpath ${COSMO_FILE})" 
echo -e "\033[32m[JOB]\033[0m WEIGHTS_DIR ........ = $(realpath ${WEIGHTS_DIR})" 
echo -e "\033[32m[JOB]\033[0m PROBE .............. = ${DATATYPE}" 
echo -e "\033[32m[JOB]\033[0m .................... " 
echo -e "\033[32m[JOB]\033[0m PYTHON ............. = $(which python)"
echo -e "\033[32m[JOB]\033[0m PYTHON VERSION ..... = $(python -V)"
echo -e "\033[32m[JOB]\033[0m SCRIPT ............. = ${PY_SCRIPT}"

# ====================== #
# RUN
# ====================== #

python "${PY_SCRIPT}" "${ARGS[@]}" 

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m All done."
exit 0
