"""
A synthetic sc-qc output (cells/<sample_id>.h5ad, the workflow contract): three samples, two
conditions, five planted cell types with their own marker genes, and a strong per-sample batch
effect (a gene-wise multiplicative factor plus sequencing depth). The truth is in
obs["annotation"] and truth.csv; the third sample lacks a few genes (genes are intersected).

  python -m tests.make_fixture data/fixture [--layout both|linked|flat]
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

TYPES = ["Neuron", "Satellite glia", "Schwann cell", "Fibroblast", "Endothelial"]
N_MARKERS = 12
N_GENES = 300
SAMPLES = [("s1", "control"), ("s2", "control"), ("s3", "pain")]
TARGET_SUM = 1e4


def genes() -> list[str]:
    return [f"G{i + 1:03d}" for i in range(N_GENES - 2)] + ["MT-CO1", "RPL3"]


def write_sample(path: Path, sample_id: str, condition: str, counts: np.ndarray, gene_names: list[str], labels) -> None:
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    n = counts.shape[0]
    obs = pd.DataFrame({"sample_id": pd.Categorical([sample_id] * n), "condition": pd.Categorical([condition] * n),
                        "subject_id": pd.Categorical([f"sub-{sample_id[-1]}"] * n), "annotation": pd.Categorical(labels)},
                       index=[f"{sample_id}:BC{i + 1:05d}-1" for i in range(n)])
    var = pd.DataFrame({"gene_ids": [f"ENSG{i:011d}" for i in range(len(gene_names))], "feature_types": "Gene Expression",
                        "mt": [g.startswith("MT-") for g in gene_names], "ribo": [g.startswith("RPL") for g in gene_names]},
                       index=gene_names)
    a = ad.AnnData(X=sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var)
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt", "ribo"], inplace=True, percent_top=None, log1p=False)
    a.obs["doublet_score"] = np.float32(0.05)
    a.obs["predicted_doublet"] = False
    a.layers["counts"] = sp.csr_matrix(counts.astype(np.float32))
    sc.pp.normalize_total(a, target_sum=TARGET_SUM)
    sc.pp.log1p(a)
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    a.uns["pennsieve"] = {"source": f"{sample_id}/sample_filtered_feature_bc_matrix.h5", "chemistry": "flex",
                          "organism": "human", "prep_version": "fixture", "qc": {"target_sum": TARGET_SUM}}
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path)


def make_synthetic(root: Path, n_cells: int = 400, seed: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    names = genes()
    base = rng.lognormal(-1.0, 1.0, N_GENES)
    profiles = np.tile(base, (len(TYPES), 1))
    for k in range(len(TYPES)):
        profiles[k, k * N_MARKERS:(k + 1) * N_MARKERS] *= 12.0
    truth = []
    for j, (sid, condition) in enumerate(SAMPLES):
        labels = np.array(TYPES)[rng.integers(len(TYPES), size=n_cells)]
        batch = rng.lognormal(0.0, 0.4, N_GENES)        # gene-wise batch effect
        depth = rng.lognormal(0.0, 0.3, n_cells)[:, None] * (1.0, 0.5, 1.5)[j]
        mean = profiles[[TYPES.index(t) for t in labels]] * batch * depth
        counts = rng.poisson(mean).astype(np.int32)
        keep = np.arange(N_GENES) if j < 2 else np.arange(N_GENES)[np.arange(N_GENES) % 50 != 49]   # s3 lacks 6 genes
        write_sample(root / "cells" / f"{sid}.h5ad", sid, condition, counts[:, keep], [names[i] for i in keep], labels)
        truth.extend({"sample_id": sid, "obs_id": f"{sid}:BC{i + 1:05d}-1", "cell_type": t} for i, t in enumerate(labels))
    with open(root / "truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "obs_id", "cell_type"])
        w.writeheader()
        w.writerows(truth)
    return root


def make_fixture(root: Path, layout: str = "both") -> dict[str, Path]:
    root = Path(root)
    out = {"linked": make_synthetic(root / "linked")}
    if layout in ("flat", "both"):
        flat = root / "flat"
        flat.mkdir(parents=True, exist_ok=True)
        for p in sorted(out["linked"].rglob("*")):
            if p.is_file():
                shutil.copy2(p, flat / p.name)
        out["flat"] = flat
    if layout == "flat":
        shutil.rmtree(out.pop("linked"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--layout", choices=["both", "linked", "flat"], default="both")
    args = ap.parse_args()
    for name, path in make_fixture(args.root, args.layout).items():
        print(f"{name}: {path}")
