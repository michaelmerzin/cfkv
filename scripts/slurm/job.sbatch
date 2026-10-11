#!/bin/bash
#SBATCH --job-name=cfy
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# usage (from the project root): sbatch scripts/slurm/job.sbatch <python script under src/> [args]
# Edit the partition / account / environment-activation lines for your cluster.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CFKV_HF_HUB=${CFKV_HF_HUB:-$HOME/.cache/huggingface/hub}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
echo "date=$(date) host=$(hostname) job=${SLURM_JOB_ID:-none} args=$*"
S="$1"; shift
python -u "src/$S" "$@"
