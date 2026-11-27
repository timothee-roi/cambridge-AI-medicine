#!/usr/bin/env bash
# Download the public resources used by Notebooks 4-6 into data/external/.
# Nothing here is MIMIC-IV data; all sources are public GitHub repositories.
#   - icd (R package, J. Wasey): HCUP multi-level CCS 2015 for ICD-9-CM + ICD-10-CM 2026 code list
#   - MedCo (Nayebi Kerdabadi et al.): CMS GEM files used by Notebook 1
#   - Mondo Disease Ontology: mondo-edit.obo (rare subset, ICD-10-CM / Orphanet / OMIM xrefs)
#   - HPO annotations: taken from the pyhpo wheel (phenotype.hpoa, genes_to_phenotype.txt, hp.obo)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/external
TMP=$(mktemp -d)
mkdir -p "$OUT/hpo"

git clone -q --depth 1 https://github.com/jackwasey/icd "$TMP/icd"
cp "$TMP/icd/data/icd9_map_multi_ccs.rda" "$TMP/icd/data/icd9_names_multi_ccs.rda" "$TMP/icd/data/icd10cm2026.rda" "$OUT/"

git clone -q --depth 1 https://github.com/mohsen-nyb/MedCo "$TMP/MedCo"
cp "$TMP/MedCo/saved_files/icd_maping/2018_I10gem.txt" "$TMP/MedCo/saved_files/icd_maping/gem_pcsi9.txt" "$OUT/"

if [ ! -d "$OUT/mondo/.git" ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/monarch-initiative/mondo "$OUT/mondo"
  git -C "$OUT/mondo" sparse-checkout set --no-cone /src/ontology/mondo-edit.obo
fi

python -m pip download --no-deps -q pyhpo==4.0.0 -d "$TMP/whl"
python - "$TMP/whl" "$OUT/hpo" <<'EOF'
import sys, zipfile, glob, os
whl = glob.glob(os.path.join(sys.argv[1], 'pyhpo-*.whl'))[0]
with zipfile.ZipFile(whl) as z:
    for name in ('hp.obo', 'phenotype.hpoa', 'genes_to_phenotype.txt'):
        with open(os.path.join(sys.argv[2], name), 'wb') as f:
            f.write(z.read(f'pyhpo/data/{name}'))
EOF
rm -rf "$TMP"
echo "done: $(ls $OUT)"
