#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --partition=prod10
#SBATCH --time=24:00:00
#SBATCH --array=0-20%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
# Whole benchmark as one job array (at most 4 running jobs, the account limit).
# From ~/cambridge_project:   mkdir -p logs && sbatch scripts/benchmark.sbatch
# Options through the environment: SEED (default 42), PROCESSED (default processed_gemfix),
# OUT (default results/benchmark/gemfix), MODELS (space-separated subset; then set --array=0-<n-1>).
# A job killed at the time limit resumes from its checkpoint when resubmitted.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/cambridge_project/venv/bin/activate
ALL=(label_prior bag_of_codes gru retain transformer transformer_pos_last transformer_setpool
     transformer_cross transformer_logitadj gram knowledge_rag_readout deep_L2 deep_L4 loop_T4
     layerloop_L2T2 modelloop_L2T2 deep_L4_adaptive loop_T4_adaptive loop_T4_gate
     layerloop_L2T2_adaptive modelloop_L2T2_adaptive)
read -r -a LIST <<< "${MODELS:-${ALL[*]}}"
M=${LIST[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID -> $M | seed ${SEED:-42} | data ${PROCESSED:-processed_gemfix} | $(hostname)"
nvidia-smi -L || true
python scripts/benchmark.py --model "$M" --seed "${SEED:-42}" \
    --processed "${PROCESSED:-processed_gemfix}" --out "${OUT:-results/benchmark/gemfix}"
