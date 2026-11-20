#!/bin/bash
#SBATCH --account=IscrC_SBIxEUCG
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sumstats
#SBATCH --output=../logs/sumstats/slurm-%x_%j.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# ====================== #
# MODULES & LIBRARIES
# ====================== #

module load python/3.10.8--gcc--8.5.0

source $HOME/dott/envs/sbi/bin/activate

set -euo pipefail

# ====================== #
# PATHS & NAMES
# 
# Expected structure:
# {RUNS_DIR}/{RUN_BASENAME}_{ID}/	(for zshells and plc modes)
# {MAINPP_DIR}/{RUN_BASENAME}_{ID}/{RICHNESS_FOLDER}/	(for richness mode)
#
# ====================== #

# Directory containing all runs
RUNS_DIR="${RUNS_DIR:-$WORK/path/to/runs}"

# Directory containing all post-processed runs' files
MAINPP_DIR="${MAINPP_DIR:-$WORK/path/to/postproc}"

# Directory containing all summary statistics
STATS_DIR="${STATS_DIR:-$WORK/stats}"

# Base name for the Pinocchio run --> (base_name)_(id)
RUN_BASENAME="${RUN_BASENAME:-model}"

# Python script for splitting the PLC in shells
PY_SCRIPT="${PY_SCRIPT:-../scripts/compute_sumstats.py}"

# ====================== #
# USER PARAMETERS
# ====================== #

# Total number of runs to use. Use "0" to use all the models in the directory
TOT_RUNS=0

# Choose between "richness", "zshells", "plc" and "plc_masscalib"
#
# richness				uses the CPP output from the richness module (could be mass calibrated)
# zshells				uses the redshift shells, bypassing the requested bins. No richness (could be mass calibrated)
# plc					uses past-light cone to compute shells at the requested bins. No richness
# plc_masscalib   		uses the mass calibrated past-light cone to compute shells at the requested bins. No richness
MODE="${MODE:-richness}"

PROFILE_MASS_CUT="${PROFILE_MASS_CUT:-1e13}"
MAX_MASS="${MAX_MASS:-1e16}"
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
MASS_VARIABLE_SPACING="${MASS_VARIABLE_SPACING-1}"

# ====================== #
# PRINTS
# ====================== #

echo -e "\033[32m[JOB]\033[0m ${SLURM_JOB_ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m RUNS_DIR ............... = ${RUNS_DIR}" 
echo -e "\033[32m[JOB]\033[0m POSTPROC_DIR ........... = ${MAINPP_DIR}" 
echo -e "\033[32m[JOB]\033[0m STATS_DIR .............. = ${STATS_DIR}" 
echo -e "\033[32m[JOB]\033[0m RUN_BASENAME ........... = ${RUN_BASENAME}" 
echo -e "\033[32m[JOB]\033[0m MODE ................... = ${MODE}" 
echo -e "\033[32m[JOB]\033[0m TOTAL RUNS ............. = ${TOT_RUNS}" 
echo -e "\033[32m[JOB]\033[0m PROFILE_MASS_CUT ....... = ${PROFILE_MASS_CUT}" 
echo -e "\033[32m[JOB]\033[0m MAX MASS ............... = ${MAX_MASS}" 
echo -e "\033[32m[JOB]\033[0m REDSHIFT_LIMIT ......... = ${REDSHIFT_LIMIT}" 
echo -e "\033[32m[JOB]\033[0m RICHNESS_MIN ........... = ${LMB_MIN}" 
echo -e "\033[32m[JOB]\033[0m RICHNESS_LIMIT ......... = ${LMB_LIMIT}" 
echo -e "\033[32m[JOB]\033[0m REDSHIFT_BINS .......... = ${Z_BINS}" 
echo -e "\033[32m[JOB]\033[0m RICHNESS_BINS .......... = ${LMB_BINS}" 
echo -e "\033[32m[JOB]\033[0m MASS_BINS .............. = ${MASS_BINS}" 
echo -e "\033[32m[JOB]\033[0m Z_LOG_SPACING .......... = ${Z_LOG_SPACING}" 
echo -e "\033[32m[JOB]\033[0m LMB_LOG_SPACING ........ = ${LMB_LOG_SPACING}" 
echo -e "\033[32m[JOB]\033[0m MASS_LOG_SPACING ....... = ${MASS_LOG_SPACING}" 
echo -e "\033[32m[JOB]\033[0m Z_VARIABLE_SPACING ..... = ${Z_VARIABLE_SPACING}" 
echo -e "\033[32m[JOB]\033[0m MASS_VARIABLE_SPACING .. = ${MASS_VARIABLE_SPACING}" 
echo -e "\033[32m[JOB]\033[0m ........................ " 
echo -e "\033[32m[JOB]\033[0m PYTHON ................. = $(which python)"
echo -e "\033[32m[JOB]\033[0m PYTHON VERSION ......... = $(python -V)"
echo -e "\033[32m[JOB]\033[0m SCRIPT ................. = ${PY_SCRIPT}"

# ====================== #
# ARGS
# ====================== #

ARGS=(
	--runs-dir "${RUNS_DIR}"
	--postproc-dir "${MAINPP_DIR}"
	--stats-dir "${STATS_DIR}"
	--mode "${MODE}"
	--base-name "${RUN_BASENAME}"
	--z-max-shell "${REDSHIFT_LIMIT}"
	--lmb-min "${LMB_MIN}"
	--lmb-max "${LMB_LIMIT}"
	--m-threshold "${PROFILE_MASS_CUT}"
	--mass-max "${MAX_MASS}"
	--z-bins "${Z_BINS}" 
	--lmb-bins "${LMB_BINS}"
	--mass-bins "${MASS_BINS}"
)

if [[ "${Z_LOG_SPACING}" == "1" ]]; then
	ARGS+=(--log-z)
fi

if [[ "${LMB_LOG_SPACING}" == "1" ]]; then
	ARGS+=(--log-lmb)
fi

if [[ "${MASS_LOG_SPACING}" == "1" ]]; then
	ARGS+=(--log-mass)
fi

if [[ "${Z_VARIABLE_SPACING}" == "1" ]]; then
	ARGS+=(--use-zknee)
fi

if [[ "${MASS_VARIABLE_SPACING}" == "1" ]]; then
	ARGS+=(--use-massknee)
fi

if [[ "${TOT_RUNS}" -gt 0 ]]; then
    ARGS+=(--total-runs ${TOT_RUNS})
fi

# ====================== #
# RUN
# ====================== #

python "${PY_SCRIPT}" "${ARGS[@]}" 

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m All done."
exit 0
