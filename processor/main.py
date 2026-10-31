"""
sc-integrate: joint integration and clustering of single-cell / single-nucleus RNA-seq samples.

Reads sc-qc's <sample_id>.h5ad files (port `cells`), chooses highly variable genes on the raw
counts across batches (seurat_v3), computes a joint PCA of the log-normalized expression,
integrates the samples with Harmony (on the PCs) or scVI (on the counts), builds the neighbour
graph, Leiden clusters and a UMAP on the integrated embedding, and writes

  cells/        the same <sample_id>.h5ad files plus obsm["X_pca"], obsm["X_integrated"],
                obsm["X_umap"], obs["cluster"], uns["pennsieve"]["integration"]; provenance.json
  labels/       labels.csv (sample_id, obs_id, cluster), cluster_summary.csv, summary.json,
                provenance.json
  integration/  metrics.csv, hvg.csv, umap_sample.png, umap_condition.png, umap_cluster.png,
                scvi_history.csv (scVI), summary.json, provenance.json

See processor/integrate.py for the method.
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from processor import threads as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("sc-integrate")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "numba", "harmonypy", "h5py"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "sc-integrate"
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "METHOD": "harmony",
    "BATCH_COLUMN": "sample_id",
    "COVARIATES": "",
    "N_HVG": "3000",
    "N_PCS": "50",
    "MAX_EPOCHS": "0",
    "N_NEIGHBORS": "15",
    "RESOLUTION": "1.0",
    "SEED": "42",
    "THREADS": "0",
}
PARAMETERS = tuple(DEFAULTS)
METHODS = ("harmony", "scvi")
COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# DOIs of the methods each run uses; the report gathers them from provenance.json.
CITATIONS = {
    "scanpy": {"doi": "10.1186/s13059-017-1382-0", "pmid": "29409532",
               "text": "Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis. "
                       "Genome Biol 2018;19:15."},
    "anndata": {"doi": "10.21105/joss.04371", "pmid": None,
                "text": "Virshup I, Rybakov S, Theis FJ, Angerer P, Wolf FA. anndata: Access and store annotated data "
                        "matrices. J Open Source Softw 2024;9(101):4371."},
    "seurat_v3 HVG": {"doi": "10.1016/j.cell.2019.05.031", "pmid": "31178118",
                      "text": "Stuart T, Butler A, Hoffman P, et al. Comprehensive Integration of Single-Cell Data. "
                              "Cell 2019;177(7):1888-1902.e21."},
    "Harmony": {"doi": "10.1038/s41592-019-0619-0", "pmid": "31740819",
                "text": "Korsunsky I, Millard N, Fan J, et al. Fast, sensitive and accurate integration of single-cell "
                        "data with Harmony. Nat Methods 2019;16(12):1289-1296."},
    "scVI": {"doi": "10.1038/s41592-018-0229-2", "pmid": "30504886",
             "text": "Lopez R, Regier J, Cole MB, Jordan MI, Yosef N. Deep generative modeling for single-cell "
                     "transcriptomics. Nat Methods 2018;15(12):1053-1058."},
    "scvi-tools": {"doi": "10.1038/s41587-021-01206-w", "pmid": "35132262",
                   "text": "Gayoso A, Lopez R, Xing G, et al. A Python library for probabilistic analysis of single-cell "
                           "omics data. Nat Biotechnol 2022;40(2):163-166."},
    "Leiden": {"doi": "10.1038/s41598-019-41695-z", "pmid": "30914743",
               "text": "Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected "
                       "communities. Sci Rep 2019;9:5233."},
    "UMAP": {"doi": "10.21105/joss.00861", "pmid": None,
             "text": "McInnes L, Healy J, Saul N, Großberger L. UMAP: Uniform Manifold Approximation and Projection. "
                     "J Open Source Softw 2018;3(29):861."},
    "scib": {"doi": "10.1038/s41592-021-01336-8", "pmid": "34949812",
             "text": "Luecken MD, Büttner M, Chaichoompu K, et al. Benchmarking atlas-level data integration in "
                     "single-cell genomics. Nat Methods 2022;19(1):41-50."},
    "silhouette": {"doi": "10.1016/0377-0427(87)90125-7", "pmid": None,
                   "text": "Rousseeuw PJ. Silhouettes: A graphical aid to the interpretation and validation of cluster "
                           "analysis. J Comput Appl Math 1987;20:53-65."},
}


class ParameterError(ValueError):
    pass


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k).strip() if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {"METHOD": raw["METHOD"].lower()}
    if p["METHOD"] not in METHODS:
        raise ParameterError(f"METHOD must be one of {', '.join(METHODS)}, got {raw['METHOD']!r}")
    p["BATCH_COLUMN"] = raw["BATCH_COLUMN"]
    p["COVARIATES"] = [c.strip() for c in raw["COVARIATES"].split(",") if c.strip()]
    for c in [p["BATCH_COLUMN"], *p["COVARIATES"]]:
        if not COLUMN.match(c):
            raise ParameterError(f"BATCH_COLUMN / COVARIATES must be obs column names ([A-Za-z_][A-Za-z0-9_]*), got {c!r}")
    if p["BATCH_COLUMN"] in p["COVARIATES"] or len(set(p["COVARIATES"])) != len(p["COVARIATES"]):
        raise ParameterError("COVARIATES must not repeat a column or BATCH_COLUMN")
    try:
        p["RESOLUTION"] = float(raw["RESOLUTION"])
        for name in ("N_HVG", "N_PCS", "MAX_EPOCHS", "N_NEIGHBORS", "SEED", "THREADS"):
            p[name] = int(raw[name])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if not (0 < p["RESOLUTION"] <= 10) or p["N_HVG"] < 50 or not (2 <= p["N_PCS"] <= 200) or p["MAX_EPOCHS"] < 0 \
            or not (2 <= p["N_NEIGHBORS"] <= 200) or p["THREADS"] < 0:
        raise ParameterError("need 0 < RESOLUTION <= 10, N_HVG >= 50, 2 <= N_PCS <= 200, MAX_EPOCHS >= 0, "
                             "2 <= N_NEIGHBORS <= 200, THREADS >= 0")
    return p


def _clean_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if hasattr(obj, "item"):   # numpy scalars
        return _clean_json(obj.item())
    return obj


def write_json(path: Path, data) -> None:
    with open(path, "w") as fh:
        json.dump(_clean_json(data), fh, indent=2)
        fh.write("\n")


def tool_versions(method: str) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    pkgs = ["scanpy", "anndata", "scikit-misc", "igraph", "umap-learn", "scikit-learn", "numpy", "scipy", "pandas",
            "h5py", "matplotlib"]
    pkgs += ["harmonypy"] if method == "harmony" else ["scvi-tools", "torch", "lightning"]
    out = {"python": platform.python_version()}
    for pkg in pkgs:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def peak_rss_mb() -> float | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except (ImportError, OSError):
        return None


def cgroup_peak_mb() -> float | None:
    try:
        return round(int(Path("/sys/fs/cgroup/memory.peak").read_text()) / (1024 * 1024), 1)
    except (OSError, ValueError):
        return None


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 2
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
    try:
        params = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    n_threads, thread_source = T.configure(params["THREADS"])
    log.info("threads: %d (%s); method %s", n_threads, thread_source, params["METHOD"])

    # numpy and friends only after the thread variables are set
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from processor import figures as F
    from processor import integrate as I
    from processor.inputs import discover

    found = discover(input_dir)
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    if not found.samples:
        log.error("no <sample_id>.h5ad found under %s", input_dir)
        return 1
    batch, covariates = params["BATCH_COLUMN"], params["COVARIATES"]
    timings: dict = {}

    def rel(p: Path) -> str:
        return str(p.relative_to(input_dir))

    # --- pass 1: counts, one sample at a time -----------------------------------------------
    t = time.monotonic()
    parts, statuses, used = [], [], []
    for s in found.samples:
        ts = time.monotonic()
        status = {"sample_id": s.sample_id, "file": rel(s.path)}
        try:
            a, where = I.read_counts(s.path, s.sample_id, [batch, *covariates])
            parts.append(a)
            used.append(s)
            status.update(status="ok", n_obs=int(a.n_obs), n_genes=int(a.n_vars), counts_from=where)
            log.info("%s: %d observations x %d genes, counts from %s, %.1fs", s.sample_id, a.n_obs, a.n_vars, where,
                     time.monotonic() - ts)
        except Exception as exc:
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log.error("%s: failed: %s", s.sample_id, exc)
        statuses.append(status)
    if not parts:
        log.error("every sample failed")
        return 1
    joint, n_dropped = I.stack(parts)
    sizes = [a.n_obs for a in parts]
    n_shared = int(joint.n_vars)
    del parts
    timings["read_counts"] = round(time.monotonic() - t, 1)
    log.info("joint counts: %d observations x %d genes in every sample (%d genes not in every sample dropped), "
             "%d non-zeros, %.1fs, peak RSS %s MB", joint.n_obs, joint.n_vars, n_dropped, joint.X.nnz,
             timings["read_counts"], peak_rss_mb())
    if joint.n_obs < 20:
        log.error("only %d observations in total; nothing to integrate", joint.n_obs)
        return 1
    batches = joint.obs[batch].astype(str).to_numpy()
    n_batches = len(set(batches))
    if n_batches < 2:
        log.warning("%s has a single value: nothing to integrate, X_integrated is the %s of the one batch",
                    batch, "PCA" if params["METHOD"] == "harmony" else "scVI latent space")

    # --- highly variable genes on the counts ------------------------------------------------
    t = time.monotonic()
    hvg, span = I.highly_variable(joint, params["N_HVG"], batch)
    genes = hvg["gene"].tolist()
    counts = joint[:, genes].X.copy() if params["METHOD"] == "scvi" else None
    obs = joint.obs
    del joint
    timings["hvg"] = round(time.monotonic() - t, 1)
    log.info("%d highly variable genes (seurat_v3 on counts, batch_key=%s, loess span %g), %.1fs", len(genes), batch, span,
             timings["hvg"])

    # --- pass 2: log-normalized X on the HVGs, joint PCA -------------------------------------
    t = time.monotonic()
    offsets = np.cumsum([0, *sizes])
    blocks = [I.read_lognorm(s.path, genes, obs.index[offsets[i]:offsets[i + 1]]) for i, s in enumerate(used)]
    X_pca = I.pca(sp.vstack(blocks, format="csr"), params["N_PCS"], params["SEED"])
    del blocks
    timings["pca"] = round(time.monotonic() - t, 1)
    log.info("PCA: %d components on the log-normalized HVGs, %.1fs", X_pca.shape[1], timings["pca"])

    # --- integration ------------------------------------------------------------------------
    t = time.monotonic()
    method_info: dict = {"method": params["METHOD"], "batch_column": batch, "covariates": covariates,
                         "n_batches": n_batches}
    history = None
    if params["METHOD"] == "harmony":
        if n_batches > 1:
            Z = I.harmony(X_pca, obs, [batch, *covariates], params["SEED"], n_threads)
        else:
            Z = X_pca.copy()
        method_info["n_dims"] = int(Z.shape[1])
    else:
        dev = I.device()
        log.info("scVI on %s", dev["device"])
        Z, scvi_info, history = I.scvi_latent(counts, obs, batch, covariates, params["MAX_EPOCHS"], params["SEED"],
                                              n_threads, dev["accelerator"])
        method_info.update(dev, **scvi_info, n_dims=int(Z.shape[1]))
        del counts
    timings["integration"] = round(time.monotonic() - t, 1)
    log.info("integration: %s, %d dimensions, %.1fs, peak RSS %s MB", params["METHOD"], Z.shape[1],
             timings["integration"], peak_rss_mb())

    # --- clusters, UMAP, metrics ------------------------------------------------------------
    t = time.monotonic()
    clusters, U, cluster_info = I.cluster_and_embed(Z, params["N_NEIGHBORS"], params["RESOLUTION"], params["SEED"])
    timings["cluster_umap"] = round(time.monotonic() - t, 1)
    log.info("%d Leiden clusters at resolution %g (neighbours %.1fs, leiden %.1fs, umap %.1fs)",
             cluster_info["n_clusters"], params["RESOLUTION"], cluster_info["neighbors_seconds"],
             cluster_info["leiden_seconds"], cluster_info["umap_seconds"])
    t = time.monotonic()
    metrics = {"unintegrated": I.metrics(X_pca, batches, clusters, params["SEED"]),
               "integrated": I.metrics(Z, batches, clusters, params["SEED"])}
    timings["metrics"] = round(time.monotonic() - t, 1)
    for k, m in metrics.items():
        log.info("%s (%s): iLISI median %.2f (scaled %.3f), cluster ASW %.3f (%d observations)", k,
                 "X_pca" if k == "unintegrated" else "X_integrated", m["ilisi_median"], m["ilisi_scaled"], m["cluster_asw"],
                 m["n_obs"])

    # --- outputs ----------------------------------------------------------------------------
    cells_dir, labels_dir, integ_dir = output_dir / "cells", output_dir / "labels", output_dir / "integration"
    for d in (cells_dir, labels_dir, integ_dir):
        d.mkdir(parents=True, exist_ok=True)
    cluster_order = sorted(set(clusters))
    record = {"app": APP_NAME, "version": app_version(), **{k: v for k, v in method_info.items() if k != "accelerator"},
              "n_hvg": len(genes), "hvg_loess_span": span, "n_pcs": int(X_pca.shape[1]), "n_neighbors": params["N_NEIGHBORS"],
              "resolution": params["RESOLUTION"], "seed": params["SEED"], "n_genes_shared": n_shared}
    t = time.monotonic()
    by_id = {s["sample_id"]: s for s in statuses}
    for i, s in enumerate(used):
        ts = time.monotonic()
        rows = np.arange(offsets[i], offsets[i + 1])
        try:
            I.write_sample(s.path, cells_dir / f"{s.sample_id}.h5ad", rows, X_pca, Z, U, clusters, cluster_order, record)
            log.info("%s: written, %d observations, %.1fs", s.sample_id, len(rows), time.monotonic() - ts)
        except Exception as exc:
            by_id[s.sample_id].update(status="failed", error=f"write: {type(exc).__name__}: {exc}")
            log.error("%s: writing failed: %s", s.sample_id, exc)
    timings["write"] = round(time.monotonic() - t, 1)

    sample_ids = obs["sample_id"].astype(str).to_numpy()
    frame = pd.DataFrame({"sample_id": sample_ids, "obs_id": obs.index.to_numpy(), "cluster": clusters})
    frame.to_csv(labels_dir / "labels.csv", index=False)
    table = pd.crosstab(frame["sample_id"], frame["cluster"])
    summary_rows = table.stack().rename("n_obs").reset_index()
    summary_rows["fraction"] = summary_rows["n_obs"] / summary_rows["sample_id"].map(table.sum(axis=1))
    summary_rows.to_csv(labels_dir / "cluster_summary.csv", index=False, float_format="%.6g")
    hvg.to_csv(integ_dir / "hvg.csv", index=False)
    pd.DataFrame([{"embedding": "X_pca" if k == "unintegrated" else "X_integrated", "metric": name, "value": m[name],
                   "n_obs": m["n_obs"]} for k, m in metrics.items() for name in ("ilisi_median", "ilisi_scaled", "cluster_asw")]
                 ).to_csv(integ_dir / "metrics.csv", index=False, float_format="%.6g")
    if history is not None:
        history.to_csv(integ_dir / "scvi_history.csv", index=False, float_format="%.6g")

    t = time.monotonic()
    figures = []
    n_note = f" ({len(frame):,} observations)"
    colourings = [("sample", sample_ids), ("cluster", clusters)]
    if "condition" in obs:
        colourings.insert(1, ("condition", obs["condition"].astype(str).to_numpy()))
    if batch not in ("sample_id", "condition"):
        colourings.insert(1, (batch, batches))
    for name, values in colourings:
        try:
            order = cluster_order if name == "cluster" else sorted(set(values))
            figures.append(F.scatter_categorical(U, values, order, f"UMAP ({params['METHOD']}) by {name}{n_note}",
                                                 integ_dir / f"umap_{name}.png", params["SEED"]).name)
        except Exception as exc:
            log.exception("figure umap_%s failed: %s", name, exc)
    timings["figures"] = round(time.monotonic() - t, 1)

    per_cluster = []
    for c in cluster_order:
        col = table[c]
        per_cluster.append({"cluster": c, "n_obs": int(col.sum()), "n_samples": int((col > 0).sum()),
                            "largest_sample_fraction": float(col.max() / col.sum())})
    n_failed = sum(1 for s in statuses if s["status"] != "ok")
    summary = {
        "n_samples": len(statuses) - n_failed, "n_failed": n_failed, "n_obs": int(len(frame)),
        "n_genes_shared": n_shared,
        "n_hvg": len(genes), "hvg_loess_span": span, "n_genes_dropped_not_in_every_sample": n_dropped, "integration": method_info,
        "n_clusters": cluster_info["n_clusters"], "clusters": per_cluster,
        "metrics": metrics, "clustering": cluster_info, "parameters": params,
    }
    cites = ["scanpy", "anndata", "seurat_v3 HVG", *(["Harmony"] if params["METHOD"] == "harmony" else ["scVI", "scvi-tools"]),
             "Leiden", "UMAP", "scib", "silhouette"]
    provenance = {
        "app": APP_NAME, "version": app_version(), "tools": tool_versions(params["METHOD"]), "parameters": params,
        "threads": {"n": n_threads, "source": thread_source}, "integration": method_info,
        "method": {
            "hvg": f"scanpy.pp.highly_variable_genes(flavor=seurat_v3, n_top_genes={params['N_HVG']}, batch_key={batch}, span={span}) on counts",
            "pca": f"scanpy.pp.pca(n_comps={params['N_PCS']}) on the log-normalized X of the HVGs (not scaled)",
            "integration": (f"harmonypy.run_harmony on X_pca, vars_use={[batch, *covariates]}" if params["METHOD"] == "harmony"
                            else f"scvi.model.SCVI(n_layers={I.N_LAYERS}, n_latent={I.N_LATENT}, gene_likelihood=nb) on the "
                                 f"HVG counts, batch_key={batch}, categorical_covariate_keys={covariates}"),
            "neighbors": f"scanpy.pp.neighbors(n_neighbors={params['N_NEIGHBORS']}, use_rep=X_integrated)",
            "clustering": f"scanpy.tl.leiden(resolution={params['RESOLUTION']}, flavor=igraph, n_iterations=2)",
            "umap": "scanpy.tl.umap on the same graph",
            "metrics": f"iLISI (inverse Simpson index of batches among {I.LISI_K} nearest neighbours, scib scaling) and "
                       f"cluster ASW (sklearn silhouette, rescaled) on at most {I.MAX_OBS_FOR_METRICS} seeded observations",
        },
        "citations": [{"name": k, **CITATIONS[k]} for k in cites],
        "inputs": {"cells": [rel(p) for p in found.seen], "skipped": found.skipped},
        "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
        "timings_seconds": {**timings, "total": round(time.monotonic() - t0, 1)},
        "peak_rss_mb": peak_rss_mb(), "cgroup_peak_mb": cgroup_peak_mb(),
        "samples": statuses, "n_ok": len(statuses) - n_failed, "n_failed": n_failed,
    }
    write_json(labels_dir / "summary.json", {k: summary[k] for k in ("n_samples", "n_obs", "n_clusters", "clusters", "parameters")})
    write_json(integ_dir / "summary.json", {**summary, "figures": figures})
    for d in (cells_dir, labels_dir, integ_dir):
        write_json(d / "provenance.json", {**provenance, "outputs": sorted(p.name for p in d.iterdir())})
    if n_failed == len(statuses):
        log.error("every sample failed")
        return 1
    log.info("done: %d samples (%d failed), %d observations, %d clusters, iLISI %.2f -> %.2f, %.1fs, peak RSS %s MB -> %s",
             len(statuses) - n_failed, n_failed, len(frame), cluster_info["n_clusters"], metrics["unintegrated"]["ilisi_median"],
             metrics["integrated"]["ilisi_median"], time.monotonic() - t0, peak_rss_mb(), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
