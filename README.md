# Next-visit diagnosis prediction for rare diseases on MIMIC-IV

Foundation-model backbones for multi-label next-visit diagnosis prediction on MIMIC-IV, and a study of why they
fail on rare labels. Pietro Liò group, Department of Computer Science and Technology, University of Cambridge.

The report is in [`report/main.pdf`](report/main.pdf) (source `report/main.tex`). Every number it contains is
traceable to [`results/measured_results.json`](results/measured_results.json) or to the saved outputs of a
notebook; experiments that have not run yet are marked as pending.

## Repository

```
notebooks/
  00_exploration.ipynb            first look at MIMIC-IV (outputs cleared)
  01_preprocessing.ipynb          cohort, code mappings (GEM, CCS, ATC), splits, samples
  02_backbone_transformer.ipynb   three modality Transformers + concatenation (MedCo backbone)
  03_looped_transformer.ipynb     weight-tied looped backbone with adaptive depth
  04_rare_disease_resources.ipynb rare diseases (Mondo, HPO) in the CCS label space; GEM audit   [public data, run]
  05_knowledge_base_rag.ipynb     retrieved knowledge documents per code; text embeddings          [documents built]
  06_knowledge_embeddings.ipynb   LLM / RAG initialised embeddings, shrinkage, knowledge readout
  07_backbone_ablations.ipynb     fusion of the three encoders (concat / sum / cross-attention), set pooling
  08_rare_code_analysis.ipynb     per-label diagnostics: frequency, recurrence, diversity, geometry, MedCo groups
src/rarepred/                     data loading, models, metrics, training, resources, knowledge retrieval
scripts/                          Slurm scripts for the HubIA DGX, download of public resources, report figures
results/                          measured results (JSON/CSV); no patient-level data
figures/, report/
```

Hypothesis A (embedding corruption), C (classifier decoupling) and the first clustering notebook were run on a
local machine and are summarised in the report; they still need to be added under `notebooks/`.

## Status (November 2026)

| experiment | notebook | status |
|---|---|---|
| Backbone | 02 | done: test sample-AUPRC 0.471, rarest quartile 0.290, most frequent 0.543 |
| Label space vs rare diseases, GEM audit | 04 | done on public data; join with MIMIC-IV frequencies runs on the cluster |
| Knowledge documents | 05 | done (589 CCS categories); text encoding runs on the cluster |
| Knowledge injection (LLM, RAG, shrinkage, readout) | 06 | implemented, tested end to end on synthetic data |
| Fusion and pooling ablations | 07 | implemented, tested end to end on synthetic data |
| Looped Transformer | 03 | corrected (dropout on the recurrent state), rerun pending |
| Per-label diagnostics on the converged model | 08 | implemented, tested end to end on synthetic data |

## Running on the HubIA DGX

```bash
# once, on a machine with internet access (public resources only)
./scripts/fetch_external.sh                  # -> data/external/
rsync -av data/external/ dgx:~/cambridge_project/data/external/

# on the DGX, from ~/cambridge_project/notebooks
mkdir -p logs executed
NB=05_knowledge_base_rag sbatch ../scripts/run_notebook.sbatch                       # writes processed/knowledge_embeddings_BioLORD-2023.npz
for c in baseline desc rag rag_shrink readout rag_readout; do
  NB=06_knowledge_embeddings RAREPRED_RUN=$c sbatch ../scripts/run_notebook.sbatch
done
NB=06_knowledge_embeddings sbatch ../scripts/run_notebook.sbatch                     # aggregate (reloads checkpoints)
for c in concat_sum sum_sum cross_sum concat_set cross_set; do
  NB=07_backbone_ablations RAREPRED_RUN=$c sbatch ../scripts/run_notebook.sbatch
done
LOOPED_RUN=L4_geometric sbatch -p prod40 ../scripts/job_looped.sbatch
NB=08_rare_code_analysis sbatch ../scripts/run_notebook.sbatch                       # needs checkpoints/backbone_ccs.pt or baseline_seed42.pt
```

Notebook 5 downloads BioLORD-2023 from Hugging Face; if the compute nodes have no internet access, download
it once on the login node (`python -c "from sentence_transformers import SentenceTransformer as S; S('FremyCompany/BioLORD-2023')"`).
At most four jobs run at once on the account (two on `prod40`/`prod80`). Notebooks read their configuration
from environment variables (`RAREPRED_RUN`, `RAREPRED_SEED`, `RAREPRED_MAX_EPOCHS`, `RAREPRED_ENCODER`,
`RAREPRED_ENCODER_TAG`, `RAREPRED_LAMBDA0`; `LOOPED_RUN`, `LOOPED_MAX_EPOCHS` for Notebook 3). A finished
configuration is reloaded from `checkpoints/` instead of retrained unless `RAREPRED_RETRAIN=1`.

To test the pipeline without MIMIC-IV, generate synthetic artifacts with the same format
(`rarepred.synthetic.make_synthetic`) into a separate `processed/` directory and point `RAREPRED_ROOT` /
`RAREPRED_PROCESSED` to it. Numbers obtained on synthetic data are meaningless and must not be reported.

## Data governance

MIMIC-IV is used under the PhysioNet credentialed data use agreement. Raw tables, processed samples,
checkpoints and predictions stay on the cluster and are excluded by `.gitignore`; notebook outputs committed
here contain aggregate statistics only. Patient records must not be sent to external LLM services; code
descriptions and the public resources of Notebooks 4 and 5 contain no patient information.

## Environment

Python 3.12, `torch` 2.7 (CUDA 12.6 on the DGX), `pyhealth==2.0.2`; see `requirements.txt`.
