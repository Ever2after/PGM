#!/bin/bash
# PGM v2 on one node (CPU work; KISTI HANGANG's aarch64 partition requires a GPU request).
# e.g. one-cell test:  [sbatch] -J pgm-v2-test scripts/slurm/v2.sbatch --models llama8b --datasets math
#      full study:     [sbatch] -J pgm-v2 scripts/slurm/v2.sbatch          (about 11 minutes)
# Stage 1 builds artifacts/v2/<model>_<dataset>.npz (skipped when present); stage 2 fits and replays every arm.
# Set PGM_PYTHON to the interpreter of the environment where pgm_asc is installed.
#SBATCH --partition=normal
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PGM_PYTHON:-python}"
"$PY" -m pgm_asc v2 prepare --workers 9 "$@"
"$PY" -m pgm_asc v2 run --workers 64 "$@"
