#!/bin/bash
#SBATCH --job-name=looped
#SBATCH --partition=prod10
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# Usage (from ~/cambridge_project/notebooks):
#   mkdir -p logs executed && LOOPED_RUN=L4_geometric sbatch ../scripts/job_looped.sbatch
# Defaults to Notebook 3's RUN list when LOOPED_RUN is not set.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/cambridge_project/venv/bin/activate

echo "Job $SLURM_JOB_ID on $(hostname) | LOOPED_RUN=${LOOPED_RUN:-<notebook default>}"
nvidia-smi -L || true
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

TAG=$(echo "${LOOPED_RUN:-default}" | tr ',' '_')
papermill 03_looped_transformer.ipynb "executed/03_looped_${TAG}_${SLURM_JOB_ID}.ipynb" \
    --log-output --request-save-on-cell-execute
