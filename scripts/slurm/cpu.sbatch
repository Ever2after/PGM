#!/bin/bash
# Run any CPU command (fit/evaluate/report) as a batch job, e.g.
#   sbatch -J fit-qwen7b-math scripts/slurm/cpu.sbatch fit qwen7b_math
#   sbatch -J evaluate-all scripts/slurm/cpu.sbatch evaluate all
# Fitting one cell (36 heads) takes about 10 minutes on 4 CPU threads; evaluating all 16 cells about 5 minutes.
# KISTI Hangang example: its aarch64 partitions require a GPU request even for CPU work.
#SBATCH --partition=normal
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4
"${PGM_PYTHON:-python}" -m pgm_asc "$@"
