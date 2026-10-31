"""
Joint integration of all samples.

A first look at every file reads its obs, genes and matrix sizes only. Pass 1 then fills each
sample's raw counts (layers["counts"], else X) into one preallocated CSR float32 matrix on the
genes present in every sample. Highly variable genes are
chosen on these counts with scanpy's seurat_v3 flavour per batch. Pass 2 reads each sample's
log-normalized X on those genes only, for the joint PCA. Then Harmony on the PCs or scVI on the
HVG counts gives obsm["X_integrated"]; neighbours, Leiden and UMAP run on it. Batch mixing (iLISI)
and cluster separation (silhouette), as scored by scib (Luecken et al. 2022), compare it with the
uncorrected PCA.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem

log = logging.getLogger("sc-integrate")

MAX_OBS_FOR_METRICS = 20_000         # the silhouette is O(n^2): seeded subsample
LISI_K = 30                          # neighbours per observation for iLISI
N_LATENT = 30                        # scVI latent dimensions
N_LAYERS = 2                         # scVI encoder/decoder hidden layers
OPTIONAL_COLUMNS = ("subject_id", "condition")   # kept when present: the auto batch key, the figure
LOESS_SPANS = (0.3, 0.5, 1.0)        # seurat_v3 loess span: Seurat's 0.3, wider when a small batch makes it singular
BYTES_PER_NNZ = 8                    # float32 value + int32 column index


@dataclass
class Meta:
    """What a sample's file holds, read without its matrices."""
    obs: pd.DataFrame        # sample_id, the batch/covariate columns asked for, OPTIONAL_COLUMNS present
    genes: pd.Index
    counts_from: str         # layers/counts or X
    nnz_counts: int
    nnz_x: int


def _counts_node(f: h5py.File) -> str:
    return "layers/counts" if "layers" in f and "counts" in f["layers"] else "X"


def _nnz(node) -> int:
    return int(node["data"].shape[0]) if isinstance(node, h5py.Group) else int(np.prod(node.shape))


def cgroup_memory_bytes() -> int | None:
    """The container's memory limit (cgroup v2 memory.max, else v1), or None."""
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(f).read_text().strip()
        except OSError:
            continue
        return int(v) if v.isdigit() and int(v) < (1 << 60) else None
    return None


def obs_columns(path: Path) -> set[str]:
    """The obs column names of a file, without reading them."""
    with h5py.File(path, "r") as f:
        return {str(c) for c in f["obs"].attrs.get("column-order", [])}


def read_meta(path: Path, sample_id: str, columns: list[str]) -> Meta:
    """obs (the columns asked for must exist and have no missing values), gene names and non-zero
    counts of one sample; refuses a matrix larger than half the container's memory."""
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        where = _counts_node(f)
        nnz_counts, nnz_x = _nnz(f[where]), _nnz(f["X"])
    if "sample_id" not in obs:
        obs["sample_id"] = sample_id
    missing = [c for c in columns if c not in obs]
    if missing:
        raise ValueError(f"obs has no column {', '.join(missing)}")
    empty = [c for c in columns if obs[c].isna().any()]
    if empty:
        raise ValueError(f"obs column {', '.join(empty)} has missing values")
    limit = cgroup_memory_bytes()
    if limit and max(nnz_counts, nnz_x) * BYTES_PER_NNZ > limit // 2:
        raise ValueError(f"{max(nnz_counts, nnz_x):,} non-zeros need more than half the {limit / 2**30:.1f} GiB memory limit")
    keep = list(dict.fromkeys(["sample_id", *columns, *(c for c in OPTIONAL_COLUMNS if c in obs)]))
    return Meta(obs[keep].copy(), pd.Index(var.index.astype(str)), where, nnz_counts, nnz_x)


def shared_genes(metas: list[Meta]) -> tuple[list[str], int]:
    """The genes present in every sample (the first sample's order) and how many others were seen."""
    shared = set(metas[0].genes)
    for m in metas[1:]:
        shared &= set(m.genes)
    genes = [g for g in metas[0].genes if g in shared]
    return genes, len(set().union(*(set(m.genes) for m in metas))) - len(genes)


def stack(paths: list[Path], metas: list[Meta], genes: list[str], counts: bool) -> sp.csr_matrix:
    """One CSR float32 matrix (samples in order, `genes` as columns) of the counts or the log-normalized
    X, filled into preallocated arrays one sample at a time, so the peak is the joint matrix plus one
    sample rather than twice the joint matrix."""
    bound = sum(m.nnz_counts if counts else m.nnz_x for m in metas)
    data = np.empty(bound, dtype=np.float32)
    indices = np.empty(bound, dtype=np.int32 if len(genes) < 2**31 else np.int64)
    indptr = [np.zeros(1, dtype=np.int64)]
    nnz = 0
    for path, m in zip(paths, metas):
        with h5py.File(path, "r") as f:
            node = f[m.counts_from if counts else "X"]
            X = read_elem(node)
            if not sp.issparse(X):
                X = sp.csr_matrix(np.asarray(X))
            obs_names = read_elem(f["obs"]).index
        if not obs_names.equals(m.obs.index):
            raise ValueError(f"{path.name}: observations changed between reads")
        X = sp.csr_matrix(X)[:, m.genes.get_indexer(genes)]
        if counts and X.nnz and not np.allclose(X.data[:10000], np.round(X.data[:10000])):
            log.warning("%s: %s is not integer-valued; seurat_v3 and scVI expect raw counts", path.stem, m.counts_from)
        data[nnz:nnz + X.nnz] = X.data
        indices[nnz:nnz + X.nnz] = X.indices
        indptr.append(X.indptr[1:].astype(np.int64) + nnz)
        nnz += X.nnz
        del X
    n_obs = sum(len(m.obs) for m in metas)
    return sp.csr_matrix((data[:nnz], indices[:nnz], np.concatenate(indptr)), shape=(n_obs, len(genes)), copy=False)


def highly_variable(joint: ad.AnnData, n_top: int, batch: str) -> tuple[pd.DataFrame, float]:
    """seurat_v3 HVGs on the counts, ranked across batches (scanpy). The mean-variance loess fit of a
    batch with few cells (~200) can be singular at Seurat's span of 0.3; a wider span is then used
    for all batches. Returns the HVG table and the span."""
    import scanpy as sc

    n_top = min(n_top, joint.n_vars)
    for span in LOESS_SPANS:
        try:
            sc.pp.highly_variable_genes(joint, flavor="seurat_v3", n_top_genes=n_top, batch_key=batch, span=span,
                                        subset=False)
            break
        except ValueError as exc:
            if span == LOESS_SPANS[-1]:
                raise
            log.warning("seurat_v3 loess fit failed at span %g (%s); retrying with a wider span", span, str(exc).strip())
    v = joint.var
    table = pd.DataFrame({"gene": v.index, "rank": v["highly_variable_rank"], "n_batches": v["highly_variable_nbatches"],
                          "highly_variable": v["highly_variable"]})
    return table[table["highly_variable"]].sort_values("rank")[["gene", "rank", "n_batches"]], span


def pca(X: sp.csr_matrix, n_pcs: int, seed: int) -> np.ndarray:
    import scanpy as sc

    a = ad.AnnData(X=X)
    n = int(min(n_pcs, X.shape[0] - 1, X.shape[1] - 1))
    sc.pp.pca(a, n_comps=n, random_state=seed)
    return np.ascontiguousarray(a.obsm["X_pca"], dtype=np.float32)


def harmony(X_pca: np.ndarray, obs: pd.DataFrame, keys: list[str], seed: int, threads: int) -> np.ndarray:
    import harmonypy

    ho = harmonypy.run_harmony(X_pca, obs[keys], keys, random_state=seed, verbose=False, ncores=max(1, threads))
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != X_pca.shape[0]:
        Z = Z.T
    return np.ascontiguousarray(Z, dtype=np.float32)


def device() -> dict:
    """Where PyTorch runs: cuda when it sees a GPU, else cpu."""
    import torch

    if torch.cuda.is_available():
        return {"accelerator": "gpu", "device": f"cuda ({torch.cuda.get_device_name(0)})", "cuda": torch.version.cuda}
    return {"accelerator": "cpu", "device": "cpu"}


def scvi_latent(counts: sp.csr_matrix, obs: pd.DataFrame, batch: str, covariates: list[str], max_epochs: int,
                seed: int, threads: int, accelerator: str) -> tuple[np.ndarray, dict, pd.DataFrame]:
    """scVI (negative binomial, 2 layers, 30 latent dimensions) on the HVG counts; returns the latent
    means, what was run, and the training history (epoch, elbo_train)."""
    import scvi
    import torch
    from lightning.pytorch.callbacks import Callback

    for name in ("scvi", "lightning", "lightning.pytorch", "lightning_fabric", "pyro"):
        logging.getLogger(name).setLevel(logging.WARNING)
    scvi.settings.verbosity = logging.WARNING
    scvi.settings.seed = seed
    torch.set_num_threads(max(1, threads))

    a = ad.AnnData(X=counts, obs=obs[[batch, *covariates]].copy())
    scvi.model.SCVI.setup_anndata(a, batch_key=batch, categorical_covariate_keys=covariates or None)
    model = scvi.model.SCVI(a, n_layers=N_LAYERS, n_latent=N_LATENT, gene_likelihood="nb")
    epochs = max_epochs or int(min(round(20000 / a.n_obs * 400), 400))   # scvi-tools' own default rule
    every = max(1, epochs // 10)

    class EpochLog(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            e = trainer.current_epoch + 1
            if e % every == 0 or e == epochs:
                elbo = trainer.callback_metrics.get("elbo_train")
                log.info("scVI: epoch %d/%d, ELBO %s", e, epochs, f"{float(elbo):.5g}" if elbo is not None else "n/a")

    t = time.monotonic()
    model.train(max_epochs=epochs, accelerator=accelerator, devices=1, enable_progress_bar=False,
                enable_model_summary=False, enable_checkpointing=False, callbacks=[EpochLog()])
    seconds = time.monotonic() - t
    Z = np.ascontiguousarray(model.get_latent_representation(), dtype=np.float32)
    h = model.history.get("elbo_train")
    history = pd.DataFrame({"epoch": np.arange(1, len(h) + 1), "elbo_train": h.iloc[:, 0].to_numpy()}) \
        if h is not None else pd.DataFrame(columns=["epoch", "elbo_train"])
    info = {"epochs": epochs, "n_latent": N_LATENT, "n_layers": N_LAYERS, "gene_likelihood": "nb",
            "train_seconds": round(seconds, 1), "seconds_per_epoch": round(seconds / epochs, 3),
            "final_elbo_train": float(history["elbo_train"].iloc[-1]) if len(history) else None}
    return Z, info, history


def cluster_and_embed(Z: np.ndarray, n_neighbors: int, resolution: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Neighbours on Z, Leiden (igraph, 2 iterations) renamed C01.. by decreasing size, UMAP."""
    import scanpy as sc

    a = ad.AnnData(X=sp.csr_matrix((Z.shape[0], 1), dtype=np.float32), obsm={"X_integrated": Z})
    info = {}
    t = time.monotonic()
    sc.pp.neighbors(a, n_neighbors=int(min(n_neighbors, Z.shape[0] - 1)), use_rep="X_integrated", random_state=seed)
    info["neighbors_seconds"] = round(time.monotonic() - t, 1)
    t = time.monotonic()
    sc.tl.leiden(a, resolution=resolution, flavor="igraph", n_iterations=2, directed=False, random_state=seed)
    info["leiden_seconds"] = round(time.monotonic() - t, 1)
    t = time.monotonic()
    sc.tl.umap(a, random_state=seed)
    info["umap_seconds"] = round(time.monotonic() - t, 1)
    raw = a.obs["leiden"].astype(str)
    counts = raw.value_counts()
    order = sorted(counts.index, key=lambda c: (-counts[c], int(c)))
    width = max(2, len(str(len(order))))
    labels = raw.map({c: f"C{i + 1:0{width}d}" for i, c in enumerate(order)}).to_numpy()
    info["n_clusters"] = len(order)
    return labels, np.ascontiguousarray(a.obsm["X_umap"], dtype=np.float32), info


def metrics(emb: np.ndarray, batch: np.ndarray, clusters: np.ndarray, seed: int) -> dict:
    """Batch mixing: iLISI, the inverse Simpson index of the batches among each observation's
    LISI_K nearest neighbours (Korsunsky et al. 2019, without its perplexity weights), median over
    observations, and scaled as in scib, (median - 1) / (batches - 1): 0 = neighbours from one
    batch, 1 = as mixed as the batch count allows. Cluster separation: the silhouette of the
    Leiden clusters rescaled to 0-1, scib's ASW. Both on one seeded subsample."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    n = emb.shape[0]
    idx = np.sort(np.random.default_rng(seed).choice(n, MAX_OBS_FOR_METRICS, replace=False)) if n > MAX_OBS_FOR_METRICS \
        else np.arange(n)
    emb, batch, clusters = emb[idx], batch[idx], clusters[idx]
    codes = pd.factorize(batch)[0]
    n_batches = int(codes.max()) + 1
    k = min(LISI_K, len(idx) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb, return_distance=False)[:, 1:]
    counts = np.zeros((len(idx), n_batches))
    np.add.at(counts, (np.repeat(np.arange(len(idx)), k), codes[nn].ravel()), 1)
    lisi = 1.0 / ((counts / k) ** 2).sum(axis=1)
    median = float(np.median(lisi))
    out = {"n_obs": int(len(idx)), "ilisi_median": median,
           "ilisi_scaled": (median - 1) / (n_batches - 1) if n_batches > 1 else float("nan"), "cluster_asw": float("nan")}
    if 1 < len(np.unique(clusters)) < len(clusters):
        out["cluster_asw"] = float((silhouette_score(emb, clusters) + 1) / 2)
    return out


def write_sample(src: Path, dst: Path, rows: np.ndarray, X_pca, Z, U, clusters, categories, record: dict) -> int:
    """The sample's file as it came in, plus obsm X_pca / X_integrated / X_umap, obs["cluster"] and
    uns["pennsieve"]["integration"]; observations keep their order."""
    a = ad.read_h5ad(src)
    a.obsm["X_pca"] = X_pca[rows]
    a.obsm["X_integrated"] = Z[rows]
    a.obsm["X_umap"] = U[rows]
    a.obs["cluster"] = pd.Categorical(clusters[rows], categories=categories)
    pen = a.uns.get("pennsieve")
    a.uns["pennsieve"] = {**(pen if isinstance(pen, dict) else {}), "integration": record}
    a.write_h5ad(dst, compression="gzip")
    return a.n_obs
