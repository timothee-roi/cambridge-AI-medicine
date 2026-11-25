#!/bin/bash
#SBATCH --job-name=rarepred
#SBATCH --partition=prod10
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# Run one notebook with papermill on the HubIA DGX. Submit from notebooks/:
#   mkdir -p logs executed
#   NB=06_knowledge_embeddings RAREPRED_RUN=desc sbatch ../scripts/run_notebook.sbatch
#   NB=07_backbone_ablations   RAREPRED_RUN=cross_sum sbatch -p prod40 ../scripts/run_notebook.sbatch
# Environment variables read by the notebooks: RAREPRED_RUN, RAREPRED_SEED, RAREPRED_MAX_EPOCHS,
# RAREPRED_RETRAIN, RAREPRED_ENCODER (Notebook 5), RAREPRED_ENCODER_TAG (Notebooks 6, 8), RAREPRED_LAMBDA0.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/cambridge_project/venv/bin/activate
echo "Job $SLURM_JOB_ID on $(hostname) | NB=$NB | RUN=${RAREPRED_RUN:-all} | SEED=${RAREPRED_SEED:-42}"
nvidia-smi -L || true
TAG=$(echo "${RAREPRED_RUN:-all}" | tr ',' '_')
papermill "${NB}.ipynb" "executed/${NB}_${TAG}_s${RAREPRED_SEED:-42}_${SLURM_JOB_ID}.ipynb" \
    --log-output --request-save-on-cell-execute
