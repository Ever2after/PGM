#!/bin/bash
# Omni in-domain cell on one whole GH200 node: collect calibration/train/dev banks (gen x2, TF, judge), fit/lock, replay test.
#   cd /scratch/t528a02/PGM-omni
#   (submit) sbatch -J omni-in-qwen7b slurm/omni_indomain.sbatch --model qwen7b
# The runner is resumable: resubmitting the same command continues an unfinished bank.
#SBATCH --job-name=omni-in
#SBATCH --partition=normal
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/t528a02/PGM-omni/slurm/logs/%x-%j.out
set -uo pipefail
ulimit -u "$(ulimit -Hu)"
cd /scratch/t528a02/PGM-omni
# Whole exclusive node: the runner assigns physical GPU indices itself per worker.
unset CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
PY=/scratch/t528a02/envs/pgm-reasc/bin/python
echo "[job $SLURM_JOB_ID $(date '+%F %T')] $(hostname): run_omni_indomain.py $*"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
$PY -u scripts/run_omni_indomain.py "$@"
code=$?
echo "[job $SLURM_JOB_ID $(date '+%F %T')] exit $code"
exit $code
