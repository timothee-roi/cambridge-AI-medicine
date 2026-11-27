#!/bin/bash
# Run the whole benchmark unattended: for each seed, submit the models that have no result yet as a job
# array (4 at a time, 4 h per task), wait for the array to finish, and resubmit the models that hit the time
# limit (they resume from their checkpoints). Then the MedCo backbone on the original labels.
#
# Start it on the login node, from ~/cambridge_project/bench, so that it survives logout:
#   nohup bash scripts/run_all.sh 12379 > logs/run_all.log 2>&1 &
# The optional argument is a job already running (e.g. the first array) to wait for before starting.
# It only sleeps and calls squeue/sbatch, so it uses no compute.
set -uo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"42 123 7"}
PASSES=${PASSES:-4}
THROTTLE=${THROTTLE:-4}
TIME=${TIME:-04:00:00}
ALL=(label_prior bag_of_codes gru retain transformer transformer_pos_last transformer_setpool
     transformer_cross transformer_logitadj gram knowledge_rag_readout deep_L2 deep_L4 loop_T4
     layerloop_L2T2 modelloop_L2T2 deep_L4_adaptive loop_T4_adaptive loop_T4_gate
     layerloop_L2T2_adaptive modelloop_L2T2_adaptive)
[ -f processed_gemfix/knowledge_embeddings_BioLORD-2023.npz ] || ALL=("${ALL[@]/knowledge_rag_readout}")

wait_for() {   # wait until no task of job $1 is in the queue
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 120; done
}

submit() {     # $1 = space-separated models, then sbatch environment; prints the job id
  local models=($1); shift
  local n=${#models[@]}
  env "$@" MODELS="${models[*]}" sbatch --parsable --array=0-$((n - 1))%"$THROTTLE" --time="$TIME" \
      scripts/benchmark.sbatch
}

[ $# -ge 1 ] && { echo "$(date) waiting for job $1"; wait_for "$1"; }

for SEED in $SEEDS; do
  for PASS in $(seq 1 "$PASSES"); do
    missing=()
    for m in "${ALL[@]}"; do
      [ -n "$m" ] && [ ! -f "results/benchmark/gemfix/${m}_s${SEED}.json" ] && missing+=("$m")
    done
    if [ ${#missing[@]} -eq 0 ]; then echo "$(date) seed $SEED complete"; break; fi
    until JOB=$(submit "${missing[*]}" SEED="$SEED"); do echo "$(date) sbatch refused, retrying"; sleep 300; done
    echo "$(date) seed $SEED pass $PASS: job $JOB, ${#missing[@]} models: ${missing[*]}"
    wait_for "$JOB"
  done
done

for SEED in $SEEDS; do
  for PASS in $(seq 1 "$PASSES"); do
    [ -f "results/benchmark/original/transformer_s${SEED}.json" ] && break
    until JOB=$(submit "transformer" SEED="$SEED" PROCESSED=processed OUT=results/benchmark/original); do sleep 300; done
    echo "$(date) original labels, seed $SEED pass $PASS: job $JOB"
    wait_for "$JOB"
  done
done
echo "$(date) all done: $(ls results/benchmark/gemfix | wc -l) result files"
