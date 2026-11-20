#!/bin/bash
#SBATCH --account=mls2026
#SBATCH --partition=a100-gpu
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=SBI_tuning
#SBATCH --output=../logs/tuning_sbi/slurm-%x_%j.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# ====================== #
# MODULES & LIBRARIES
# ====================== #

#module load python/3.10.8--gcc--8.5.0

#source $WORK/Code/envs/sbi/bin/activate
#source $HOME/MLS2026/mchiaren/sbi/bin/activate
source $HOME/dott/envs/sbi/bin/activate

#set -euo pipefail

# ====================== #
# PATHS & NAMES
# ====================== #

# Where to create the optuna dir and logs
MAIN_DIR="${MAIN_DIR:-../}"

# Path to the summary statistics dir
SUMSTATS_DIR="${SUMSTATS_DIR:-$WORK/stats}"

# Cosmo Param file
COSMO_FILE="${COSMO_FILE:-../SobolSeq/models_parameters_3dim.txt}"

# Python script
PY_SCRIPT="${PY_SCRIPT:-../scripts/tune.py}"

# Optuna study name
STUDY_NAME="${STUDY_NAME:-npe_tune}"

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

DATATYPE="${DATATYPE:-counts_mass_z}"
ENSEMBLE=1
NUM_ENSEMBLE=5
VAL_FRAC=0.05
TEST_FRAC=0.1
PCA=0
EMBEDDING_NET=1
WHICH_EMBED="FC"
N_TRIALS=150
# K=0 to not use K-fold crossvalidation
K_CROSSVAL=0

# ====================== #
# ARGS (edit)
# ====================== #

ARGS=(
	--directory "$(realpath ${MAIN_DIR})"
	--study-name "${STUDY_NAME}"
	--dataset-type "${DATATYPE}"
	--parameters-path "${COSMO_FILE}"
	--sumstats-path "${SUMSTATS_DIR}"
	--val-frac "${VAL_FRAC}"
	--test-frac "${TEST_FRAC}"
	--n-ensemble "${NUM_ENSEMBLE}"
	--k-crossval "${K_CROSSVAL}"
	--which-param "${PARAMS[@]}"
	--which-embedding "${WHICH_EMBED}"
	--n-trials "${N_TRIALS}"
	--density-est maf nsf
	--hidden-features 32 64 96 128 256
	--batchsize 64 128 256
	--num-transforms-max 15
	--learning-rate-min 1e-6
	--learning-rate-max 1e-3
	--cnn-hidden-features 32 64 128
	--fc-hidden-features 64 128 256
	--fc-out-dimension 16 32 48
	--fc-layers-max 6
)

# do not edit here
if [ "$ENSEMBLE" -eq 1]; then
	ARGS+=(--ensemble-training)
fi

if [ "$PCA" -eq 1 ]; then
	ARGS+=(--do-pca)
fi

if [ "$EMBEDDING_NET" -eq 1 ]; then
	ARGS+=(--use-embedding)
fi

# ====================== #
# PRINTS
# ====================== #

echo -e "\033[32m[JOB]\033[0m ${SLURM_JOB_ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m MAIN_DIR ........... = $(realpath ${MAIN_DIR})" 
echo -e "\033[32m[JOB]\033[0m SUMSTATS_DIR ....... = ${SUMSTATS_DIR}" 
echo -e "\033[32m[JOB]\033[0m COSMO_FILE ......... = $(realpath ${COSMO_FILE})" 
echo -e "\033[32m[JOB]\033[0m DATASET TYPE ....... = ${DATATYPE}" 
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
