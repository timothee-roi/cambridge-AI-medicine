"""Text descriptions of codes, retrieval of external knowledge, text encoders.

Nothing here reads MIMIC-IV records: descriptions come from the code systems
(CCS, ATC) and retrieved knowledge from Mondo and the HPO annotations, so the
documents (and their embeddings) can be built and inspected anywhere, and code
descriptions may be sent to an external LLM without breaching the MIMIC-IV
data use agreement.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .resources import (ccs_label, ccs_path_labels, icd10_prefix_to_ccs, mondo_disease_keys)


# ---------------------------------------------------------------------------
# Descriptions from the code systems
# ---------------------------------------------------------------------------
def describe_ccs(code: str, labels: dict) -> str:
    path = ccs_path_labels(code, labels)
    return " > ".join(path) if path else f"CCS {code}"


def pyhealth_names(system: str, codes):
    """Names from PyHealth's code tables (ATC, CCSPROC). Falls back to the code."""
    try:
        from pyhealth.medcode import InnerMap
        im = InnerMap.load(system)
    except Exception:
        return {c: c for c in codes}
    out = {}
    for c in codes:
        try:
            out[c] = im.lookup(c)
        except Exception:
            out[c] = c
    return out


# ---------------------------------------------------------------------------
# Retrieval: Mondo diseases + HPO phenotypes + genes attached to a CCS code
# ---------------------------------------------------------------------------
class CCSKnowledgeIndex:
    """Structured retrieval from external resources for each CCS category.

    A Mondo disease is attached to CCS category c when one of its ICD-10-CM
    cross-references maps to c through the same GEM + CCS pipeline as
    Notebook 1 (``icd10_prefix_to_ccs``).
    """

    def __init__(self, mondo, gem, icd9_to_ccs, phenotypes, genes, hpo_names, rare_only=False):
        self.mondo, self.phen, self.genes, self.hpo_names = mondo, phenotypes, genes, hpo_names
        keys = sorted(gem)
        self.by_ccs = defaultdict(list)
        for mid, t in mondo.items():
            if rare_only and "rare" not in t.subsets:
                continue
            cats = set()
            for c in t.icd10cm_exact | t.icd10cm_related:
                cats |= icd10_prefix_to_ccs(c, gem, icd9_to_ccs, keys)
            for c in cats:
                self.by_ccs[c].append(mid)

    def disease_profile(self, mid):
        t = self.mondo[mid]
        ph, gn = set(), set()
        for k in mondo_disease_keys(t):
            ph |= self.phen.get(k, set())
            gn |= self.genes.get(k, set())
        return ph, gn

    def retrieve(self, ccs_code, k_diseases=8, k_phen=12, k_genes=10):
        mids = self.by_ccs.get(ccs_code, [])
        prof = {m: self.disease_profile(m) for m in mids}
        # rank: rare diseases with the richest phenotype annotation first
        ranked = sorted(mids, key=lambda m: (-("rare" in self.mondo[m].subsets),
                                             -len(prof[m][0]), self.mondo[m].name))
        phen_cnt, gene_cnt = Counter(), Counter()
        for m in mids:
            phen_cnt.update(prof[m][0]); gene_cnt.update(prof[m][1])
        return {
            "n_diseases": len(mids),
            "n_rare": sum("rare" in self.mondo[m].subsets for m in mids),
            "diseases": [(m, self.mondo[m].name, self.mondo[m].definition) for m in ranked[:k_diseases]],
            "phenotypes": [self.hpo_names.get(h, h) for h, _ in phen_cnt.most_common(k_phen)],
            "genes": [g for g, _ in gene_cnt.most_common(k_genes)],
        }

    def document(self, ccs_code, label_text, max_chars=1500, **kw):
        r = self.retrieve(ccs_code, **kw)
        parts = [f"Diagnosis category: {label_text}."]
        if r["diseases"]:
            parts.append("Includes conditions such as " + "; ".join(n for _, n, _ in r["diseases"]) + ".")
        if r["phenotypes"]:
            parts.append("Typical clinical features: " + ", ".join(r["phenotypes"]) + ".")
        if r["genes"]:
            parts.append("Associated genes: " + ", ".join(r["genes"]) + ".")
        for _, name, d in r["diseases"][:2]:
            if d:
                parts.append(f"{name}: {d}")
        return " ".join(parts)[:max_chars], r


# ---------------------------------------------------------------------------
# Text encoders
# ---------------------------------------------------------------------------
def encode_texts(texts, model_name="FremyCompany/BioLORD-2023", device=None, batch_size=64):
    """Sentence embeddings, L2-normalised.

    ``model_name='tfidf'`` gives a TF-IDF + SVD encoder: it is used only to test
    the pipeline offline and must not be reported as an LLM embedding.
    """
    if model_name.startswith("tfidf"):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(min_df=1, ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
        k = max(2, min(128, X.shape[1] - 1, X.shape[0] - 1))
        Z = TruncatedSVD(k, random_state=0).fit_transform(X)
    else:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(model_name, device=device)
        Z = enc.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                       convert_to_numpy=True)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z.astype(np.float32)


def embedding_matrix(code2idx: dict, vectors: dict, dim: int):
    """Stack vectors in vocabulary order; row 0 = PAD (zeros). Missing codes get
    zeros and are reported."""
    n = max(code2idx.values()) + 1
    E = np.zeros((n, dim), dtype=np.float32)
    missing = []
    for c, i in code2idx.items():
        v = vectors.get(c)
        if v is None:
            missing.append(c)
        else:
            E[i] = v
    return E, missing
