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

# Main output directory
DIR="${DIR:-../}"

# Base name for the Pinocchio run --> (base_name)_(id)
RUN_BASENAME="${RUN_BASENAME:-model}"

# Directory containing all summary statistics
STATS_DIR="${STATS_DIR:-$WORK/stats}"

# Cosmo Param file
COSMO_FILE="${COSMO_FILE:-../SobolSeq/models_parameters_3dim.txt}"

# Python script for training the neural network
PY_SCRIPT="${PY_SCRIPT:-../scripts/neuralnet.py}"

# ====================== #
# CATALOGUE PARAMETERS
# ====================== #

MODE="plc_mock"

PROFILE_MASS_CUT="${PROFILE_MASS_CUT:-1e13}"
REDSHIFT_MIN="${REDSHIFT_MIN:-0}"
REDSHIFT_LIMIT="${REDSHIFT_LIMIT:-2.}"
LMB_MIN="${LMB_MIN:-5}"
LMB_LIMIT="${LMB_LIMIT:-400}"

Z_BINS="${Z_BINS:-10}"
LMB_BINS="${LMB_BINS:-10}"
MASS_BINS="${MASS_BINS:-10}"

Z_LOG_SPACING="${Z_LOG_SPACING:-0}"
LMB_LOG_SPACING="${LMB_LOG_SPACING:-0}"
MASS_LOG_SPACING="${MASS_LOG_SPACING:-1}"

Z_VARIABLE_SPACING="${Z_VARIABLE_SPACING:-1}"
MASS_VARIABLE_SPACING="${MASS_VARIABLE_SPACING:-1}"

DO_PCA="0"
NORM_INPUTS="1"

TUNE="0"

# ====================== #
# ARCHITECTURE
# ====================== #

HIDDEN_NODES="532"
NLAYERS="4"
DROPOUT="0.015685146221465336"

# ====================== #
# OPTIM. HYPERPARAMETERS
# ====================== #

# Dataset splitting
VAL_FRAC="0.05"
TEST_FRAC="0.1"

# Shared
NEPOCHS="400"
BATCHSIZE="128"
USE_SCHEDULER="1"

# Training
LR="0.0008309627205944977"
WDECAY="6.562127824656945e-05"
BETA1="0.8811552278986757"
BETA2="0.9543452658073587"
EPS="2.5957107476692074e-08"
SCHEDULER_FACT="0.25347924512608777"

# ====================== #
# DATASET DIR
# ====================== #

M_CUT_FMT=$(awk -v v="$PROFILE_MASS_CUT" 'BEGIN { printf "%.1e", v }' | sed 's/+//')
Z_MAX_FMT=$(awk -v v="$REDSHIFT_LIMIT" 'BEGIN { printf "%.3f", v }')
Z_MIN_FMT=$(awk -v v="$REDSHIFT_MIN"   'BEGIN { printf "%.3f", v }')

if [ "$Z_LOG_SPACING" -eq 1 ] && [ "$Z_VARIABLE_SPACING" -eq 1 ]; then
    Z_SPACING="logvar"
elif [ "$Z_LOG_SPACING" -eq 1 ]; then
    Z_SPACING="log"
elif [ "$Z_VARIABLE_SPACING" -eq 1 ]; then
    Z_SPACING="linvar"
else
    Z_SPACING="lin"
fi

if [ "$MASS_LOG_SPACING" -eq 1 ] && [ "$MASS_VARIABLE_SPACING" -eq 1 ]; then
    MASS_SPACING="logvar"
elif [ "$MASS_LOG_SPACING" -eq 1 ]; then
    MASS_SPACING="log"
elif [ "$MASS_VARIABLE_SPACING" -eq 1 ]; then
    MASS_SPACING="linvar"
else
    MASS_SPACING="lin"
fi

if [ "$LMB_LOG_SPACING" -eq 1 ]; then
    LMB_SPACING="log"
else
    LMB_SPACING="lin"
fi

STATS_DIRNAME="${MODE}_m${M_CUT_FMT}_z${Z_MAX_FMT}-${Z_MIN_FMT}_${Z_SPACING}-zbins${Z_BINS}_${MASS_SPACING}-massbins${MASS_BINS}"
if [ "$MODE" = "richness" ]; then
	STATS_DIRNAME+="_${LMB_SPACING}-lmbbins${LMB_BINS}"
fi

STATS_FULLPATH="${STATS_DIR}/${STATS_DIRNAME}"

# ====================== #
# ARGS
# ====================== #

ARGS=(
	--directory "${DIR}"
	--stats_path "${STATS_FULLPATH}"
	--params_path "${COSMO_FILE}"
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

if [[ "${DO_PCA}" == "1" ]]; then
	ARGS+=(--do_pca)
fi

if [[ "${NORM_INPUTS}" == "1" ]]; then
	ARGS+=(--normalise_inputs)
fi
# ====================== #
# PRINTS
# ====================== #

echo -e "\033[32m[JOB]\033[0m ${SLURM_JOB_ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m STATS_DIR .............. = ${STATS_DIR}" 
echo -e "\033[32m[JOB]\033[0m DATASET ................ = ${STATS_DIRNAME}" 
echo -e "\033[32m[JOB]\033[0m COSMO_FILE ............. = ${COSMO_FILE}" 
echo -e "\033[32m[JOB]\033[0m RUN_BASENAME ........... = ${RUN_BASENAME}" 
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
