#!/bin/bash
# Build a response bank end to end on one 4-GPU node, then fit and evaluate:
#   sbatch -J qwen7b-math scripts/slurm/collect.sbatch --model qwen7b --dataset math
# Add --memory 0.5 for Llama-3.1-8B on Omni-MATH. Resubmitting the same command resumes.
# KISTI Hangang example; adapt partition/comment/time to your cluster.
#SBATCH --partition=normal
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset CUDA_VISIBLE_DEVICES
"${PGM_PYTHON:-python}" -m pgm_asc collect run "$@"
