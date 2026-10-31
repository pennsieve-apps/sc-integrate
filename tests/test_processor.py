"""The synthetic fixture through the processor (both input layouts, Harmony and scVI), the
contract of the rewritten .h5ad files, skipped samples and parameter handling. No Docker."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from processor import main as app
from tests.make_fixture import make_fixture


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> dict[str, Path]:
    return make_fixture(tmp_path_factory.mktemp("fixture"), "both")


def run(input_dir: Path, output_dir: Path, monkeypatch, **params) -> int:
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/matplotlib")
    for name in app.PARAMETERS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("THREADS", "2")
    for name, value in params.items():
        monkeypatch.setenv(name, str(value))
    return app.main()


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def ari_vs_truth(out: Path, root: Path) -> float:
    from sklearn.metrics import adjusted_rand_score

    truth_file = root / "truth.csv" if (root / "truth.csv").exists() else root.parent / "linked" / "truth.csv"
    truth = {r["obs_id"]: r["cell_type"] for r in rows(truth_file)}
    labels = rows(out / "labels" / "labels.csv")
    return adjusted_rand_score([truth[r["obs_id"]] for r in labels], [r["cluster"] for r in labels])


@pytest.mark.parametrize("layout", ["linked", "flat"])
def test_harmony(fixture_root, tmp_path, monkeypatch, layout):
    import anndata as ad
    import numpy as np

    out = tmp_path / "out"
    assert run(fixture_root[layout], out, monkeypatch, RESOLUTION=0.3) == 0
    assert sorted(p.name for p in out.iterdir()) == ["cells", "integration", "labels"]
    assert {p.name for p in (out / "cells").iterdir()} == {"s1.h5ad", "s2.h5ad", "s3.h5ad", "provenance.json"}
    assert {"labels.csv", "cluster_summary.csv", "summary.json", "provenance.json"} <= {p.name for p in (out / "labels").iterdir()}
    assert {"metrics.csv", "hvg.csv", "umap_sample.png", "umap_condition.png", "umap_cluster.png", "summary.json",
            "provenance.json"} <= {p.name for p in (out / "integration").iterdir()}

    src = ad.read_h5ad(fixture_root["linked"] / "cells" / "s3.h5ad")
    a = ad.read_h5ad(out / "cells" / "s3.h5ad")
    assert list(a.obs_names) == list(src.obs_names) and list(a.var_names) == list(src.var_names)
    assert (a.layers["counts"] != src.layers["counts"]).nnz == 0
    assert a.obsm["X_pca"].shape == (400, 50) and a.obsm["X_integrated"].shape == (400, 50)
    assert a.obsm["X_umap"].shape == (400, 2)
    assert all(c.startswith("C") for c in a.obs["cluster"].cat.categories)
    assert a.uns["pennsieve"]["integration"]["method"] == "harmony"
    assert a.uns["pennsieve"]["integration"]["batch_column"] == "subject_id"   # BATCH_COLUMN=auto
    assert a.uns["pennsieve"]["prep_version"] == "fixture"   # earlier keys kept

    labels = rows(out / "labels" / "labels.csv")
    assert list(labels[0]) == ["sample_id", "obs_id", "cluster"] and len(labels) == 1200
    assert ari_vs_truth(out, fixture_root[layout]) >= 0.9

    summary = json.loads((out / "integration" / "summary.json").read_text())
    assert summary["n_samples"] == 3 and summary["n_obs"] == 1200
    assert summary["n_genes_shared"] == 294 and summary["n_genes_dropped_not_in_every_sample"] == 6
    m = summary["metrics"]
    assert m["unintegrated"]["ilisi_scaled"] < 0.2 < 0.4 < m["integrated"]["ilisi_scaled"]   # the planted batch effect is removed
    assert np.isfinite(m["integrated"]["cluster_asw"])

    prov = json.loads((out / "integration" / "provenance.json").read_text())
    assert prov["app"] == "sc-integrate" and prov["version"] == "0.1.0" and prov["n_ok"] == 3
    assert {"app", "version", "tools", "parameters", "inputs", "started", "finished", "samples", "peak_rss_mb",
            "citations"} <= set(prov)
    assert sorted(Path(p).name for p in prov["inputs"]) == ["s1.h5ad", "s2.h5ad", "s3.h5ad"]
    dois = {c["doi"] for c in prov["citations"]}
    assert {"10.1038/s41592-019-0619-0", "10.1038/s41592-021-01336-8", "10.1038/s41598-019-41695-z"} <= dois
    assert all(set(c) <= {"doi", "pmid", "text"} and c["text"] for c in prov["citations"])


def test_scvi(fixture_root, tmp_path, monkeypatch):
    import anndata as ad

    out = tmp_path / "out"
    assert run(fixture_root["linked"], out, monkeypatch, METHOD="scvi", MAX_EPOCHS=100, COVARIATES="condition",
               BATCH_COLUMN="sample_id",
               RESOLUTION=0.3) == 0
    a = ad.read_h5ad(out / "cells" / "s1.h5ad")
    assert a.obsm["X_integrated"].shape == (400, 30)
    rec = a.uns["pennsieve"]["integration"]
    assert rec["method"] == "scvi" and rec["epochs"] == 100 and list(rec["covariates"]) == ["condition"]
    assert len(rows(out / "integration" / "scvi_history.csv")) == 100
    summary = json.loads((out / "integration" / "summary.json").read_text())
    assert summary["integration"]["accelerator"] in ("cpu", "gpu")
    assert ari_vs_truth(out, fixture_root["linked"]) >= 0.9
    m = summary["metrics"]
    assert m["integrated"]["ilisi_scaled"] > m["unintegrated"]["ilisi_scaled"] + 0.3
    prov = json.loads((out / "integration" / "provenance.json").read_text())
    dois = {c["doi"] for c in prov["citations"]}
    assert {"10.1038/s41592-018-0229-2", "10.1038/s41587-021-01206-w"} <= dois and "10.1038/s41592-019-0619-0" not in dois
    assert rec["batch_column"] == "sample_id"


def test_auto_batch_without_subject_id(fixture_root, tmp_path, monkeypatch):
    import anndata as ad

    inp = tmp_path / "in" / "cells"
    inp.mkdir(parents=True)
    for p in sorted((fixture_root["linked"] / "cells").glob("*.h5ad")):
        a = ad.read_h5ad(p)
        del a.obs["subject_id"]
        a.write_h5ad(inp / p.name)
    out = tmp_path / "out"
    assert run(inp.parent, out, monkeypatch) == 0
    assert json.loads((out / "integration" / "summary.json").read_text())["integration"]["batch_column"] == "sample_id"


def test_bad_sample_is_skipped(fixture_root, tmp_path, monkeypatch):
    inp = tmp_path / "in" / "cells"
    inp.mkdir(parents=True)
    for p in (fixture_root["linked"] / "cells").iterdir():
        (inp / p.name).symlink_to(p)
    (inp / "broken.h5ad").write_bytes(b"not hdf5")
    (inp / "bad name.h5ad").write_bytes(b"x")
    out = tmp_path / "out"
    assert run(inp.parent, out, monkeypatch) == 0
    prov = json.loads((out / "labels" / "provenance.json").read_text())
    assert prov["n_ok"] == 3 and prov["n_failed"] == 1
    assert "cells/bad name.h5ad" in prov["inputs_skipped"]
    assert not (out / "cells" / "broken.h5ad").exists()


def test_links_outside_input_are_skipped(fixture_root, tmp_path, monkeypatch):
    inp = tmp_path / "in"
    (inp / "cells" / "nested").mkdir(parents=True)
    for p in (fixture_root["linked"] / "cells").iterdir():
        (inp / "cells" / p.name).symlink_to(p)   # INPUT_DIR's own folder: allowed
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "x9.h5ad").write_bytes(b"x")
    (inp / "cells" / "nested" / "x9.h5ad").symlink_to(elsewhere / "x9.h5ad")
    out = tmp_path / "out"
    assert run(inp, out, monkeypatch) == 0
    prov = json.loads((out / "labels" / "provenance.json").read_text())
    assert prov["inputs_skipped"] == {"cells/nested/x9.h5ad": "links outside INPUT_DIR"}


def test_missing_batch_column_fails(fixture_root, tmp_path, monkeypatch):
    assert run(fixture_root["linked"], tmp_path / "out", monkeypatch, BATCH_COLUMN="lane") == 1


def test_no_input(tmp_path, monkeypatch):
    (tmp_path / "in").mkdir()
    assert run(tmp_path / "in", tmp_path / "out", monkeypatch) == 1


@pytest.mark.parametrize("name,value", [("METHOD", "combat"), ("RESOLUTION", "0"), ("N_PCS", "1"), ("N_HVG", "10"),
                                        ("BATCH_COLUMN", "a;b"), ("COVARIATES", "subject_id"), ("MAX_EPOCHS", "-1"),
                                        ("SEED", "x")])
def test_bad_parameters(fixture_root, tmp_path, monkeypatch, name, value):
    assert run(fixture_root["linked"], tmp_path / "out", monkeypatch, **{name: value}) == 2
