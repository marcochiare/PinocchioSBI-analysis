#!/bin/bash
#SBATCH --account=IscrC_SBIxEUCG
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=neuralnet
#SBATCH --output=../logs/neuralnet/slurm-%x_%j.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# ====================== #
# MODULES & LIBRARIES
# ====================== #

#module load python/3.10.8--gcc--8.5.0

#source $HOME/dott/envs/sbi/bin/activate
source $WORK/Code/envs/sbi/bin/activate

#set -euo pipefail

# ====================== #
# PATHS & NAMES
# ====================== #

# Where to save the model weights 
MAIN_DIR="${MAIN_DIR:-../}"

# Directory containing all summary statistics
SUMSTATS_DIR="${SUMSTATS_DIR:-$WORK/stats}"

# Name of the study, if tuning
STUDY_NAME="${STUDY_NAME:-emulator_tuning}"

# Python script for training the neural network
PY_SCRIPT="${PY_SCRIPT:-../scripts/emulator.py}"

# ====================== #
# DATASET SETUP
# ====================== #

DATATYPE="${DATATYPE:-counts_lmb_z}"

NORM_INPUTS=0
NORM_OUTPUTS=1
TYPE_NORMOUT="log"

TUNE=1

# ====================== #
# ARCHITECTURE
# ====================== #

HIDDEN_NODES="64"
NLAYERS="4"
DROPOUT="0.015685146221465336"
INCREASE_NODES="1"

# ====================== #
# OPTIM. HYPERPARAMETERS
# ====================== #

# Dataset splitting
VAL_FRAC="0.05"
TEST_FRAC="0.1"

# Shared
NEPOCHS="400"
BATCHSIZE="512"
USE_SCHEDULER="1"

# Training
LR="0.0008309627205944977"
WDECAY="6.562127824656945e-05"
BETA1="0.8811552278986757"
BETA2="0.9543452658073587"
EPS="2.5957107476692074e-08"
SCHEDULER_FACT="0.25347924512608777"

# ====================== #
# ARGS
#
# no need to edit from here
# ====================== #

ARGS=(
	--directory "${MAIN_DIR}"
	--numbercounts_path "${SUMSTATS_DIR}"
	--probe "${DATATYPE}"
	--type_norm "${TYPE_NORMOUT}"
	--study_name "${STUDY_NAME}"
	--val_frac "${VAL_FRAC}"
	--test_frac "${TEST_FRAC}"
	--hidden_nodes "${HIDDEN_NODES}"
	--nlayers "${NLAYERS}"
	--dropout "${DROPOUT}"
	--learning_rate "${LR}"
	--batchsize "${BATCHSIZE}"
	--weight_decay "${WDECAY}"
	--beta1	"${BETA1}"
	--beta2 "${BETA2}"
	--eps "${EPS}"
	--num_epochs "${NEPOCHS}"
	--scheduler_factor "${SCHEDULER_FACT}"
)

if [[ "${TUNE}" == "1" ]]; then
	ARGS+=(--tune)
fi

if [[ "${USE_SCHEDULER}" == "0" ]]; then
	ARGS+=(--no_scheduler)
fi

if [[ "${NORM_INPUTS}" == "1" ]]; then
	ARGS+=(--normalise_inputs)
fi

if [[ "${NORM_OUTPUTS}" == "1" ]]; then
	ARGS+=(--normalise_outputs)
fi

if [[ "${INCREASE_NODES}" == "1" ]]; then
	ARGS+=(--increase_nodes)
fi

# ====================== #
# PRINTS
# ====================== #

echo -e "\033[32m[JOB]\033[0m ${SLURM_JOB_ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m MAIN_DIR ............... = $(realpath ${MAIN_DIR})" 
echo -e "\033[32m[JOB]\033[0m SUMSTATS_DIR ........... = $(realpath ${SUMSTATS_DIR})" 
echo -e "\033[32m[JOB]\033[0m PROBE .................. = ${DATATYPE}" 
echo -e "\033[32m[JOB]\033[0m ........................ " 
echo -e "\033[32m[JOB]\033[0m PYTHON ................. = $(which python)"
echo -e "\033[32m[JOB]\033[0m PYTHON VERSION ......... = $(python -V)"
echo -e "\033[32m[JOB]\033[0m SCRIPT ................. = ${PY_SCRIPT}"

# ====================== #
# RUN
# ====================== #

python "${PY_SCRIPT}" "${ARGS[@]}" 

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m All done."
exit 0
