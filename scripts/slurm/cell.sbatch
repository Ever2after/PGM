#!/bin/bash
# Run one follow-up cell end to end on a whole 4-GPU node (KISTI Hangang GH200 example):
# fitting banks, fit/lock, evaluation banks, replay. Submit from the repository root:
#   (submit) PGM_PYTHON=/path/to/.venv-reasc/bin/python sbatch -J llama8b-math scripts/slurm/cell.sbatch \
#            --config configs/pgm_followup/llama8b_math.json
# Resumable: resubmitting the same command continues an unfinished bank. Adjust partition,
# --comment and resources for your cluster.
#SBATCH --job-name=cell
#SBATCH --partition=normal
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/slurm/%x-%j.out
set -uo pipefail
ulimit -u "$(ulimit -Hu)"
cd "${SLURM_SUBMIT_DIR:-.}"
# Whole exclusive node: the runner assigns physical GPU indices itself per worker.
unset CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
PY=${PGM_PYTHON:-python}
echo "[job ${SLURM_JOB_ID:-local} $(date '+%F %T')] $(hostname): run_cell.py $*"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
$PY -u scripts/run_cell.py "$@"
code=$?
echo "[job ${SLURM_JOB_ID:-local} $(date '+%F %T')] exit $code"
exit $code
