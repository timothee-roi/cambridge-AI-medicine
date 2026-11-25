"""Public code systems and rare-disease resources.

Everything in this module uses public data only (no MIMIC-IV records):

* HCUP multi-level CCS for ICD-9-CM (2015 release), as redistributed in the
  ``icd`` R package (``icd9_map_multi_ccs.rda`` / ``icd9_names_multi_ccs.rda``).
* CMS General Equivalence Mapping ICD-10-CM -> ICD-9-CM (2018 ``I10gem``),
  the same file used by Notebook 1 (copied from the MedCo repository).
* Mondo Disease Ontology (``mondo-edit.obo``), for the rare-disease subset and
  its cross-references to ICD-10-CM, Orphanet and OMIM.
* HPO annotations (``phenotype.hpoa``, ``genes_to_phenotype.txt``).

The CCS "LVL 4 EXT" convention follows Notebook 1 exactly: an ICD-9-CM code is
mapped to its deepest non-empty multi-level category (LVL4, else LVL3, ...).
"""
from __future__ import annotations

import bisect
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CCS multi-level (ICD-9-CM)
# ---------------------------------------------------------------------------
LEVELS = ["lvl1", "lvl2", "lvl3", "lvl4"]


def _sym(t):
    while t is not None and t.info.type.name == "REF":
        t = t.referenced_object
    if t is not None and t.info.type.name == "SYM":
        return t.value.value.decode()
    return None


def _attr(obj, name):
    node = obj.attributes
    while node is not None:
        if _sym(node.tag) == name:
            return node.value[0]
        node = node.value[1] if len(node.value) > 1 else None
    return None


def _strs(obj):
    return [x.value.decode() if x.value is not None else None for x in obj.value]


def load_multilevel_ccs_from_rda(icd_r_data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read multi-level CCS from the ``icd`` R package data directory.

    Returns
    -------
    table : DataFrame with one row per ICD-9-CM code and columns lvl1..lvl4
            (empty string when the level does not exist) and ``ccs_ext``.
    labels : dict level -> {category code -> label}
    """
    import rdata  # local import: only needed for this reader

    d = Path(icd_r_data_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = rdata.parser.parse_file(d / "icd9_map_multi_ccs.rda").object.value[0]
        pn = rdata.parser.parse_file(d / "icd9_names_multi_ccs.rda").object.value[0]

    code_to = {lvl: {} for lvl in LEVELS}
    labels = {}
    for li, lvl in enumerate(LEVELS):
        m = pm.value[li]
        dn = _attr(m, "dimnames")
        cats = _strs(dn.value[0]) if dn is not None else _strs(_attr(m, "names"))
        for cat, codes in zip(cats, (_strs(v) for v in m.value)):
            for c in codes:
                code_to[lvl][c] = (cat or "").strip()
        nm = pn.value[li]
        nn = _attr(nm, "names")
        labels[lvl] = dict(zip(_strs(nn), _strs(nm))) if nn is not None else {}

    rows = []
    for code in code_to["lvl1"]:
        row = {"icd9": code}
        for lvl in LEVELS:
            row[lvl] = code_to[lvl].get(code, "")
        row["ccs_ext"] = next((row[l] for l in reversed(LEVELS) if row[l]), "")
        rows.append(row)
    return pd.DataFrame(rows), labels


def load_multilevel_ccs_from_hcup_csv(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read the original HCUP ``ccs_multi_dx_tool_2015.csv`` (used on the cluster)."""
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [c.strip().strip("'").strip() for c in raw.columns]
    raw = raw.apply(lambda s: s.str.strip().str.strip("'").str.strip())
    rows, labels = [], {lvl: {} for lvl in LEVELS}
    for _, r in raw.iterrows():
        row = {"icd9": r["ICD-9-CM CODE"]}
        for i, lvl in enumerate(LEVELS, start=1):
            code = (r.get(f"CCS LVL {i}") or "").strip()
            row[lvl] = code
            if code:
                labels[lvl][code] = r.get(f"CCS LVL {i} LABEL", "")
        row["ccs_ext"] = next((row[l] for l in reversed(LEVELS) if row[l]), "")
        rows.append(row)
    return pd.DataFrame(rows), labels


def ccs_label(code: str, labels: dict) -> str:
    """Human-readable label for a CCS LVL-4-EXT code (any depth)."""
    depth = code.count(".") + 1
    lab = labels.get(f"lvl{depth}", {}).get(code, "")
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", lab or "").strip()


def ccs_path_labels(code: str, labels: dict) -> list[str]:
    """Labels of all ancestors, from chapter to the category itself."""
    parts = code.split(".")
    out = []
    for i in range(1, len(parts) + 1):
        out.append(ccs_label(".".join(parts[:i]), labels))
    return [o for o in out if o]


# ---------------------------------------------------------------------------
# GEM ICD-10-CM -> ICD-9-CM (Notebook 1 rule: first mapping wins)
# ---------------------------------------------------------------------------
def load_gem_first(path: str | Path) -> dict[str, str]:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["icd10", "icd9", "flag"], dtype=str)
    mapping: dict[str, str] = {}
    for a, b in zip(df["icd10"], df["icd9"]):
        a, b = str(a).strip(), str(b).strip()
        if a not in mapping:
            mapping[a] = b
    return mapping


def icd10_prefix_to_ccs(icd10_code: str, gem: dict, icd9_to_ccs: dict,
                        _sorted_keys: list | None = None) -> set[str]:
    """Map an ICD-10-CM code (possibly a non-billable parent such as ``Q87``)
    to the set of CCS categories reached by its billable descendants.

    A billable code maps through the GEM exactly as in Notebook 1; a parent code
    is expanded to every GEM key that starts with it.
    """
    code = icd10_code.replace(".", "").strip().upper()
    if code in gem:
        keys = [code]
    else:
        sk = _sorted_keys if _sorted_keys is not None else sorted(gem)
        i = bisect.bisect_left(sk, code)
        keys = []
        while i < len(sk) and sk[i].startswith(code):
            keys.append(sk[i])
            i += 1
    out = set()
    for k in keys:
        icd9 = gem.get(k)
        if icd9 and icd9 != "NoDx":
            c = icd9_to_ccs.get(icd9)
            if c:
                out.add(c)
    return out


# ---------------------------------------------------------------------------
# Mondo
# ---------------------------------------------------------------------------
@dataclass
class MondoTerm:
    id: str
    name: str = ""
    definition: str = ""
    subsets: set = field(default_factory=set)
    icd10cm_exact: set = field(default_factory=set)
    icd10cm_related: set = field(default_factory=set)
    orphanet: set = field(default_factory=set)
    omim: set = field(default_factory=set)
    synonyms: list = field(default_factory=list)
    obsolete: bool = False


_XREF = re.compile(r"^xref:\s*(\S+)(?:\s*\{(.*)\})?")


def parse_mondo(obo_path: str | Path) -> dict[str, MondoTerm]:
    terms: dict[str, MondoTerm] = {}
    cur: MondoTerm | None = None
    in_term = False
    with open(obo_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("["):
                if cur is not None:
                    terms[cur.id] = cur
                in_term = line == "[Term]"
                cur = None
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                cur = MondoTerm(id=line[4:].strip())
            elif cur is None:
                continue
            elif line.startswith("name: "):
                cur.name = line[6:].strip()
            elif line.startswith("def: "):
                m = re.match(r'def:\s*"(.*)"\s*\[', line)
                cur.definition = m.group(1) if m else ""
            elif line.startswith("subset: "):
                cur.subsets.add(line.split()[1])
            elif line.startswith("synonym: "):
                m = re.match(r'synonym:\s*"(.*?)"\s+EXACT', line)
                if m:
                    cur.synonyms.append(m.group(1))
            elif line.startswith("is_obsolete: true"):
                cur.obsolete = True
            elif line.startswith("xref: "):
                m = _XREF.match(line)
                if not m:
                    continue
                ref, ann = m.group(1), m.group(2) or ""
                if ref.startswith("ICD10CM:"):
                    code = ref.split(":", 1)[1]
                    if "MONDO:equivalentTo" in ann:
                        cur.icd10cm_exact.add(code)
                    else:
                        cur.icd10cm_related.add(code)
                elif ref.startswith("Orphanet:"):
                    cur.orphanet.add(ref.split(":", 1)[1])
                elif ref.startswith("OMIM:"):
                    cur.omim.add(ref.split(":", 1)[1])
    if cur is not None:
        terms[cur.id] = cur
    return {k: v for k, v in terms.items() if not v.obsolete and k.startswith("MONDO:")}


# ---------------------------------------------------------------------------
# HPO annotations
# ---------------------------------------------------------------------------
def load_hpo_annotations(hpoa_path: str | Path, genes_path: str | Path | None = None,
                         hpo_obo_path: str | Path | None = None):
    """Return (phenotypes, genes, hpo_names).

    phenotypes : dict disease_id ('OMIM:xxx' / 'ORPHA:xxx') -> set of HPO ids (aspect P)
    genes      : dict disease_id -> set of gene symbols
    hpo_names  : dict HPO id -> name
    """
    ann = pd.read_csv(hpoa_path, sep="\t", comment="#", dtype=str)
    ann = ann[ann["aspect"] == "P"]
    ann = ann[ann["qualifier"].isna() | (ann["qualifier"] != "NOT")]
    phen = ann.groupby("database_id")["hpo_id"].apply(set).to_dict()
    genes = {}
    if genes_path is not None:
        g = pd.read_csv(genes_path, sep="\t", dtype=str)
        genes = g.groupby("disease_id")["gene_symbol"].apply(set).to_dict()
    names = {}
    if hpo_obo_path is not None:
        cur = None
        with open(hpo_obo_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("id: HP:"):
                    cur = line[4:].strip()
                elif line.startswith("name: ") and cur:
                    names[cur] = line[6:].strip()
                    cur = None
    return phen, genes, names


def mondo_disease_keys(t: MondoTerm) -> list[str]:
    return [f"ORPHA:{o}" for o in sorted(t.orphanet)] + [f"OMIM:{o}" for o in sorted(t.omim)]
