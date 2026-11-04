#!/bin/bash
# Run resume_omni.py on one whole GH200 node (4 GPUs: gen shard0, gen shard1, TF features, xVerify judge).
#   cd /scratch/t528a02/PGM-omni
#   (submit) sbatch -J omni-smoke -t 01:00:00 slurm/omni.sbatch run --model qwen7b --gpus 0,1,2,3 --offline --limit 2
#   (submit) sbatch -J omni-qwen7b slurm/omni.sbatch run --model qwen7b --gpus 0,1,2,3 --offline
# The runner is resumable: resubmitting the same command continues an unfinished bank.
#SBATCH --job-name=omni
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
echo "[job $SLURM_JOB_ID $(date '+%F %T')] $(hostname): resume_omni.py $*"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
$PY -u resume_omni.py "$@"
code=$?
echo "[job $SLURM_JOB_ID $(date '+%F %T')] exit $code"
exit $code
