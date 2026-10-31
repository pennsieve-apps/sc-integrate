# sc-integrate

Joint **integration and clustering** of single-cell and single-nucleus RNA-seq samples (scRNA-seq,
snRNA-seq, 10x 3'/5'/Flex, the RNA half of multiome). The third step of the *Single-nucleus Atlas
Mapping & Condition Comparison* workflow
(`sc-prep → sc-qc → sc-integrate → sc-annotate → {sc-markers, st-compare} → sc-report`).

**For a biologist.** When nuclei from many donors, surgeries or sequencing runs are analysed
together, the same cell type from different samples often lands in different places, because each
sample carries its own technical signature (a "batch effect"). This app removes those signatures so
that, for example, satellite glia from all twelve DRG donors form one group, while keeping the
differences between cell types. It then groups the nuclei into clusters (`C01`, `C02`, ... largest
first; the same name means the same cluster in every sample), draws a UMAP map of all nuclei
coloured by sample, condition and cluster, and reports how well the samples are mixed. The next
step (sc-annotate) names the clusters.

**For a computational analyst.** All samples are stacked on the genes present in every sample.
Highly variable genes (HVGs) are chosen on the raw counts with scanpy's `seurat_v3` flavour per
batch (`batch_key=BATCH_COLUMN`; Stuart et al. 2019). A joint PCA is computed on the log-normalized
`X` of those HVGs (not scaled). Then:

- `METHOD=harmony` (the cpu variant's default): Harmony (Korsunsky et al. 2019, `harmonypy`)
  corrects the PCs for `BATCH_COLUMN` and any `COVARIATES`.
- `METHOD=scvi` (the gpu variant's default): scVI (Lopez et al. 2018, scvi-tools) is trained on the
  HVG counts with `batch_key=BATCH_COLUMN` and `categorical_covariate_keys=COVARIATES`
  (negative binomial, 2 hidden layers, 30 latent dimensions); its latent means are the embedding.

`scanpy.pp.neighbors` and `scanpy.tl.leiden(flavor="igraph", n_iterations=2)` run on the
integrated embedding (`obsm["X_integrated"]`), followed by `scanpy.tl.umap`. Two metrics from the
scib benchmark (Luecken et al. 2022) compare the integrated embedding with the uncorrected PCA.

Why these choices. The scib benchmark (Luecken et al. 2022) found that HVG selection improves
integration and that scaling pushes methods towards batch removal at the cost of biological
variation, so the PCA here is on HVGs and unscaled; it also found that scANVI, Scanorama, scVI and
scGen perform well, particularly on complex integration tasks. The integration chapter of the
single-cell best-practices book (Zappia, Lücken, Kim, Heumos; sc-best-practices.org, cited as
Heumos et al. 2023) selects HVGs batch-aware with `highly_variable_genes(batch_key=...)`, as here,
and summarises the benchmarks as Harmony and Seurat performing well for simple batch correction
and scVI, scGen, scANVI and Scanorama for complex integration. scVI needs no labels and scales to
atlases on a GPU, so it is the gpu variant's default. Harmony needs no GPU and corrects 92,000
nuclei in about a second on CPU (the neighbour graph and UMAP take longer), so it is the cpu
default.

## Method

1. **Inputs.** Every `<sample_id>.h5ad` in `cells/` (or anywhere under the input without links).
   The batches are `BATCH_COLUMN`; `auto` takes `subject_id`, the donor key sc-prep writes on every
   cell, when every sample has it, else `sample_id` (logged and recorded).
   `BATCH_COLUMN` and `COVARIATES` are checked against the first sample's `obs` before anything is
   read (exit 2 when missing). Every file is checked with `safe_inputs` (the platform's symlink
   rule; no HDF5 external links, external storage or virtual datasets). A file that cannot be read,
   lacks a requested column or has missing values in it, or whose matrix would need more than half
   the container's memory limit is logged and skipped; the run fails (exit 1) only when no sample
   is usable.
2. **Counts, pass 1.** A first look at each file reads its `obs`, genes and matrix sizes only. Then
   `layers["counts"]` (else `X`) of each sample, one at a time, is copied into one preallocated CSR
   float32 matrix on the genes in every sample, so the peak is the joint matrix plus one sample
   (the number of genes dropped is logged and in `integration/summary.json`).
3. **HVGs.** `scanpy.pp.highly_variable_genes(flavor="seurat_v3", n_top_genes=N_HVG,
   batch_key=BATCH_COLUMN)`. Seurat's loess span of 0.3 can be singular for a batch of ~200
   nuclei; the app then retries with span 0.5, then 1.0, for all batches, logs it and records the
   span in `integration/summary.json` (the DRG demo needs 0.5).
4. **PCA, pass 2.** The log-normalized `X` of each sample on the HVGs, stacked the same way, then
   `scanpy.pp.pca` (`N_PCS`, seeded); written as `obsm["X_pca"]`.
5. **Integration** (`METHOD`, above) → `obsm["X_integrated"]`. With one batch only, Harmony is
   skipped (`X_integrated` = `X_pca`), with a warning. scVI trains for `MAX_EPOCHS` (0 =
   scvi-tools' own rule, `min(400, 20000 / n_cells × 400)`) on the GPU when PyTorch sees one
   (`torch.cuda.is_available()`, logged), else on the CPU. In the gpu image (`TORCH_FLAVOUR=cuda`)
   a missing GPU is an error (exit 1, logged with torch's CUDA version) before anything is read,
   so a gpu run never falls back to hours on CPU; with a GPU, the device, CUDA version and NVIDIA
   driver are logged.
6. **Clusters and UMAP.** Neighbours (`N_NEIGHBORS`) on `X_integrated`, Leiden (`RESOLUTION`),
   renamed `C01..` by decreasing size; UMAP of every nucleus → `obsm["X_umap"]`.
7. **Metrics** on one seeded subsample of at most 20,000 nuclei, for `X_pca` and `X_integrated`:
   - **iLISI** (batch mixing): for each nucleus, the inverse Simpson index of the batches among its
     30 nearest neighbours (the LISI of Korsunsky et al. 2019, with uniform instead of
     perplexity-weighted neighbours); `ilisi_median` is the effective number of batches around a
     typical nucleus, `ilisi_scaled` = (median − 1) / (batches − 1) as in scib (0 = neighbours from
     one batch, 1 = fully mixed; unequal batch sizes keep it below 1 even when mixing is perfect).
   - **Cluster ASW**: the silhouette (Rousseeuw 1987) of the Leiden clusters, rescaled to 0–1 as
     scib does: how compact and separated the clusters are in that embedding.
8. **Outputs.** Each sample's file is rewritten as it came in (all slots kept, observations in the
   same order) plus `obsm["X_pca"]`, `obsm["X_integrated"]`, `obsm["X_umap"]`, `obs["cluster"]`
   (categorical, the same categories in every sample) and `uns["pennsieve"]["integration"]`
   (method, batch column, covariates, HVG count and loess span, PCs, neighbours, resolution, seed,
   device and epochs for scVI).

Deviations from the single-cell best-practices book (Heumos et al. 2023 and sc-best-practices.org):

- **HVGs by seurat_v3, not deviance.** The book's feature-selection chapter recommends deviance on
  raw counts (computed there with the R package scry). This app uses scanpy's
  seurat_v3 flavour on the counts per batch, the batch-aware selection of the book's integration
  chapter, because it is in the Python stack (no R in the image) and works per batch.
- **One method per run.** The book suggests running several integration methods and comparing them
  with the scIB metrics; this app runs one per invocation and reports two of scib's metrics (iLISI
  and the cluster silhouette), not the full suite (scib-metrics needs JAX, which would double the
  image for two numbers computed here with scikit-learn). Run it twice (`METHOD=harmony`,
  `METHOD=scvi`) to compare.

## Ports

| Port | Direction | Content |
|---|---|---|
| `cells` | input, required | sc-qc's `<sample_id>.h5ad` (log-normalized `X`, `layers["counts"]`, `obs["sample_id"]` and sample metadata) |
| `cells` | output | `cells/<sample_id>.h5ad` with the embeddings and clusters added; `provenance.json` |
| `labels` | output | `labels/labels.csv` and summaries |
| `integration` | output | `integration/` metrics, HVGs, figures |

The input uses `placement: subfolder`; without links every `.h5ad` under `INPUT_DIR` is a sample.
Inputs are listed by the shared `processor/safe_inputs.py` (vendored unchanged from
workflow-single-cell): the platform's links (the entries of `INPUT_DIR` and of a real
`INPUT_DIR/cells/` folder) are followed one hop; an upstream entry that is itself a link elsewhere,
or any deeper link that leaves the input, is skipped and logged.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `METHOD` | `harmony` (gpu variant: `scvi`) | `harmony` (PCs) or `scvi` (counts). |
| `BATCH_COLUMN` | `auto` | `obs` column of the batches (also the HVG batches). `auto`: `subject_id` (the donor) when every sample has it, else `sample_id`. By donor, one donor's samples (left/right ganglia, several joint tissues) keep their differences; `sample_id` treats every library as a batch. |
| `COVARIATES` | empty | Comma-separated extra categorical `obs` columns to remove (e.g. `chemistry`). Never the condition you want to compare. |
| `N_HVG` | `3000` | Highly variable genes (50–20,000; capped by the shared genes). |
| `N_PCS` | `50` | PCA components (2–200). |
| `MAX_EPOCHS` | `0` | scVI epochs; 0 = scvi-tools' rule (400 below 20,000 nuclei). Ignored by Harmony. |
| `N_NEIGHBORS` | `15` | Neighbours in the graph (2–200). |
| `RESOLUTION` | `1.0` | Leiden resolution (0–10]; higher = more clusters. |
| `SEED` | `42` | Seed for PCA, Harmony, scVI, Leiden, UMAP and the metric subsample. |
| `THREADS` | `0` | BLAS / numba / Harmony / PyTorch threads; 0 = the container's CPU quota (else the CPUs available). |

Bad values exit with code 2 (as does an `INPUT_DIR` that is not a folder).

## Outputs

`cells/`: `<sample_id>.h5ad` (above), `provenance.json`.

`labels/`:

| File | Columns / content |
|---|---|
| `labels.csv` | `sample_id, obs_id, cluster` for every nucleus of every sample whose file was written |
| `cluster_summary.csv` | `sample_id, cluster, n_obs, fraction` (fraction of the sample), the same samples |
| `provenance.json` | app version, tool versions, parameters, method, citations (DOIs), inputs seen, timings, per-sample status, peak RSS |

`integration/`:

| File | Columns / content |
|---|---|
| `metrics.csv` | `embedding` (`X_pca`, `X_integrated`), `metric` (`ilisi_median`, `ilisi_scaled`, `cluster_asw`), `value`, `n_obs` |
| `hvg.csv` | `gene, rank, n_batches` (seurat_v3 rank across batches; batches in which the gene was highly variable) |
| `umap_sample.png`, `umap_condition.png` (when `obs["condition"]` exists), `umap_cluster.png`, `umap_<BATCH_COLUMN>.png` (when it is not `sample_id`) | UMAP of all nuclei, drawn in a seeded random order so no sample hides another |
| `scvi_history.csv` | `epoch, elbo_train` (scVI only) |
| `summary.json` | samples, nuclei, clusters with `n_obs`, `n_samples` (samples present) and `largest_sample_fraction` (a cluster made of one sample is a residual batch effect or a sample-specific population), genes shared/dropped, HVGs and loess span, integration details (device, epochs, seconds per epoch, final ELBO), metrics, clustering timings, figures, parameters |
| `provenance.json` | as above |

## Results on the demo data

sc-qc output of Discover dataset 475 (C2 DRG, 10x Flex, 12 donors, one sample each; sc-prep
capped samples at 3,000 nuclei), sc-qc run of 2026-10-31 14:37 EDT:
`data/work/drg-c2/integrate/` (the input of sc-annotate), cpu image, defaults (Harmony,
`BATCH_COLUMN=auto` → `subject_id`, 3,000 HVGs, 50 PCs, resolution 1.0).

- 18,438 nuclei × 17,681 genes in every sample (819 genes not in every sample dropped: eight
  samples have 18,082 features, four have 18,099). The loess fit of the HVG step needs the wider
  span (0.5); the smallest sample has 165 nuclei.
- **Mixing** (median iLISI, effective number of donors among a nucleus's 30 neighbours): 1.42 on
  the PCA, **2.84** after Harmony (scaled 0.038 → 0.167). Cluster ASW 0.546 → 0.550.
- **15 clusters**, 8–12 donors each. By the mean log-normalized expression of canonical genes:
  SNAP25-high C06, C09, C12 (neuronal; C04 and C14 resemble them with lower SNAP25); FABP7/APOE-high
  C01, C05, C11 (satellite glia); DCN-high C02, C07 (fibroblasts); PECAM1/VWF C08 (endothelium);
  CD163 C10 (macrophages); MPZ/PRX C13 (myelinating Schwann cells); RGS5/ACTA2 C15 (pericytes /
  smooth muscle); C03 is low in every marker. sc-annotate names them.
- **Two groups of donors differ in composition.** The four 18,099-feature samples (CC0032, CC0037,
  CC0048, CC0050; 10,900 nuclei) put 0–6 % of their nuclei in the neuronal-like clusters
  (C04, C06, C09, C12, C14), the other eight 39–88 %; the fibroblast, endothelial and macrophage
  clusters (C02, C07, C08, C10) come almost only from the four (5,439 of 5,582 nuclei). The
  satellite-glia clusters (C01, C05, C11) hold nuclei from both groups, so their glia do share
  clusters after integration. This looks like tissue content (how much ganglion vs root, capsule
  or vessels was sampled) more than a technical batch, but the feature counts also suggest a
  different probe set or pipeline version for the four; check with the markers (sc-markers) and the
  paper before comparing conditions across the two groups.
- **scVI** on the same data (cpu image, 20 epochs only, to time it): iLISI 1.42 → 2.07, 19
  clusters. With scvi-tools' default epochs (400 at this size) on an earlier stand-in for sc-qc of
  the same samples (20,204 nuclei, light filtering; native arm64): iLISI 1.47 → **3.00** (Harmony on
  the same stand-in: 2.83), cluster ASW 0.547, 21 clusters: slightly more mixing, not a different
  picture.

On the synthetic fixture (5 planted types, 3 samples with a planted gene-wise batch effect), both
methods recover the planted types exactly (ARI 1.0 at resolution 0.3) and raise the scaled iLISI
from < 0.2 to > 0.4.

## Runtime

`computeTypes: [standard]` (cpu variant), 4 vCPU / 16 GiB, 8 h timeout; gpu variant:
`computeTypes: [gpu]`, 14 GiB. Measured in the linux/amd64 images under emulation on an M-series
Mac (Docker, `--cpus 4`, `--memory 4g`/`4500m`; native x86 is faster):

| Input | Method | Wall time | Peak RSS (cgroup peak) |
|---|---|---|---|
| DRG demo, 12 samples, 18,438 nuclei, 23.5 M non-zero counts | Harmony | 138 s (HVG 13 s, PCA 2 s, Harmony 0.5 s, neighbours 91 s, UMAP 9 s, write 9 s) | 2.22 GB (2.52 GB) |
| same | scVI, 20 epochs | 220 s (3.9 s per epoch) | 2.49 GB (2.69 GB) |
| DRG demo ×5 (binomially thinned copies of an earlier sc-qc run, 60 samples), 92,195 nuclei, 98.8 M non-zeros | Harmony | 221 s (neighbours 89 s, UMAP 42 s) | 2.42 GB (3.50 GB) |

**Memory.** Counts and HVGs are filled into preallocated arrays one sample at a time. Up to and
including HVG selection the peak is about **0.5 GB + 14 bytes per non-zero count** (measured: 0.84
GB at 23.5 M non-zeros, 1.90 GB at 98.8 M; the joint counts take 8 bytes per non-zero, scanpy's
seurat_v3 statistics most of the rest). After that, the neighbour graph, UMAP and writing peak at
about 2.2 GB plus ~3.4 KB per nucleus (2.22 GB at 18 k, 2.42 GB at 92 k nuclei). These are the
cells-per-GiB figures: ~70 M non-zeros per GiB, i.e. ~65,000 nuclei per GiB at ~1,100 detected
genes per nucleus or ~28,000 at 2,500. **Estimates** extrapolated from these two runs (not run):
300 k nuclei × 30 k genes with 3,000 detected genes each (900 M non-zeros) need ~13 GB, inside the
16 GiB default (the convention's target is 30 GiB); scVI adds a copy of the HVG counts (8 bytes per
HVG non-zero) and PyTorch (~0.3 GB here), so the gpu tier's 14 GiB should hold ~200 k nuclei at
2,500 genes each.

**Time.** Harmony itself takes seconds; the neighbour graph and UMAP dominate (~minutes per 100 k
nuclei). scVI on CPU: 3.9 s per epoch for 18,438 nuclei here (emulated; 1.6 s per epoch for 20,204
nuclei native arm64 on 4 threads), and scvi-tools' default epochs fall with size (400 below 20 k
nuclei, 27 at 300 k), so a CPU scVI run is ~25 min at either end (estimate); on a GPU it should be
far less (not measured: no local GPU).

**GPU image.** torch 2.14.1+cu126 (the platform's GPU nodes run NVIDIA driver 550, CUDA 12.4;
CUDA 13 builds need driver ≥ 580). Built locally; imports torch (CUDA 12.6), scvi-tools 1.5.2,
lightning 2.6.6 and harmonypy; without a GPU it stops at start with an error (exit 1), as intended.

Images (unpacked, `du` inside the container): cpu 1.44 GB, gpu 6.97 GB.

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python       # pytest on the synthetic fixture (Harmony and scVI), no Docker
make run PYTHON=.venv/bin/python        # cpu image over data/fixture/linked -> data/output/{cells,labels,integration}
make image-gpu                          # the gpu variant's image (CUDA PyTorch)
make validate PYTHON=.venv/bin/python   # app.yml (jsonschema, Draft 2020-12) and CITATION.cff (pipx run cffconvert==2.0.0;
                                        # set CFFCONVERT=... to use another install: it needs jsonschema < 4)
```

By hand, on sc-qc output:

```sh
docker build --platform linux/amd64 --build-arg TORCH_FLAVOUR=cpu -t sc-integrate:dev .
docker run --rm --platform linux/amd64 --memory 4g --network none --read-only --tmpfs /tmp \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output -e METHOD=harmony \
  -v /path/to/sc-qc/output/cells:/data/input/cells:ro -v "$PWD/data/output:/data/output" sc-integrate:dev
```

## Limitations

- **Integration can remove biology.** Whatever differs systematically between batches is treated
  as technical. When conditions are confounded with batches (each condition sequenced separately,
  or one sample per donor and condition), real condition effects in cell states can be reduced.
  Never put the condition in `COVARIATES`; compare conditions downstream on the counts (st-compare
  does pseudobulk on `layers["counts"]`, not on the embedding).
- **The embedding is for neighbours, clusters and UMAPs only.** `X_integrated` does not correct the
  expression values; differential expression must use the counts with the batch in the model.
- **The metrics have no ground truth.** Without cell-type labels, scib's biology-conservation
  metrics (cell-type ASW, NMI/ARI against labels) cannot be computed; the cluster ASW is measured
  against clusters found on the integrated embedding itself, so it favours `X_integrated`. A high
  iLISI with merged cell types is over-correction: check the UMAP by cluster and the markers
  (sc-markers) before trusting the clusters.
- **Composition differences look like poor mixing.** A sample rich in a cell type others lack (e.g.
  one surgical DRG with many neurons) keeps its own region, and its `ilisi` stays low there; that is
  correct, not a failure.
- **Harmony vs scVI.** Harmony corrects a linear PCA and can under-correct complex, non-linear batch
  effects (different chemistries, nuclei vs cells); scVI handles these better but is stochastic,
  needs many epochs on CPU, and its result changes slightly with the hardware.
- **Clusters are not cell types** and their number depends on `RESOLUTION`; rare populations (< ~1 %)
  may merge into neighbours.
- **Genes not measured in every sample are dropped** for the joint analysis (logged); the rewritten
  files keep all their own genes.

## Licences

Our code: Apache-2.0 (`LICENSE`). The image bundles and imports **harmonypy** (GPL-3.0-or-later)
and **igraph** (GPL-2.0-or-later), so the image as a whole is distributed under the GPL-3.0;
everything else is BSD/MIT/Apache/MPL/PSF, and the gpu image adds NVIDIA's redistributable CUDA
libraries. Details in `THIRD_PARTY.md`. The demo data are CC BY 4.0.

## Credits

- **scanpy** (F. Alexander Wolf, Philipp Angerer, Fabian J. Theis and the scverse team): HVG
  selection, PCA, neighbour graph, Leiden and UMAP.
- **anndata** (Isaac Virshup, Sergei Rybakov, Fabian J. Theis, Philipp Angerer, F. Alexander Wolf;
  scverse): reading and writing the `.h5ad` files.
- **Seurat v3 HVG method** (Tim Stuart, Rahul Satija and colleagues, New York Genome Center), as
  implemented in scanpy with **scikit-misc**'s LOESS (Hassan Kibirige): the batch-aware variable
  genes.
- **Harmony** (Ilya Korsunsky, Soumya Raychaudhuri and colleagues; Python port **harmonypy** by Kamil
  Slowikowski): `METHOD=harmony`; its LISI metric is the basis of the iLISI reported here.
- **scVI** (Romain Lopez, Jeffrey Regier, Michael B. Cole, Michael I. Jordan, Nir Yosef) and
  **scvi-tools** (Adam Gayoso, Romain Lopez, Nir Yosef and colleagues; scverse): `METHOD=scvi`.
- **PyTorch** (Adam Paszke, Soumith Chintala and colleagues) and **Lightning** (Lightning AI): the
  deep-learning stack under scvi-tools.
- **Leiden** (Vincent A. Traag, Ludo Waltman, Nees Jan van Eck), via **python-igraph** (Gábor Csárdi,
  Tamás Nepusz and the igraph team): the clusters.
- **UMAP** (Leland McInnes, John Healy, Nathaniel Saul, Lukas Großberger): the 2-D maps.
- **scib** (Malte D. Luecken, Fabian J. Theis and colleagues): the benchmark that informed the method
  choices and the metric definitions.
- **Silhouette** (Peter J. Rousseeuw), computed with **scikit-learn** (Fabian Pedregosa and
  colleagues): the cluster ASW.
- **Single-cell best practices** (Lukas Heumos, Fabian J. Theis and the Single-cell Best Practices
  Consortium): the workflow this app follows.
- **Demo data**: Asta Arendt-Tranholm, Ishwarya Sankaranarayanan and Theodore J. Price shared the
  single-nucleus C2 DRG data on Pennsieve Discover (dataset 475); described in Arendt-Tranholm et
  al., Brain 2026.

## Citations

Verified on 2026-10-31 against Crossref and PubMed (articles: title, authors, year, journal,
volume, issue, pages, DOI, PMID), DataCite (preprints, the dataset) and the Pennsieve Discover API
(dataset 475); full author lists in `CITATION.cff`.

- Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis.
  *Genome Biol* 2018;19(1):15. doi:10.1186/s13059-017-1382-0. PMID 29409532.
- Virshup I, Rybakov S, Theis FJ, Angerer P, Wolf FA. anndata: Access and store annotated data
  matrices. *J Open Source Softw* 2024;9(101):4371. doi:10.21105/joss.04371.
- Stuart T, Butler A, Hoffman P, et al. Comprehensive Integration of Single-Cell Data. *Cell*
  2019;177(7):1888-1902.e21. doi:10.1016/j.cell.2019.05.031. PMID 31178118.
- Korsunsky I, Millard N, Fan J, et al. Fast, sensitive and accurate integration of single-cell data
  with Harmony. *Nat Methods* 2019;16(12):1289-1296. doi:10.1038/s41592-019-0619-0. PMID 31740819.
- Lopez R, Regier J, Cole MB, Jordan MI, Yosef N. Deep generative modeling for single-cell
  transcriptomics. *Nat Methods* 2018;15(12):1053-1058. doi:10.1038/s41592-018-0229-2. PMID 30504886.
- Gayoso A, Lopez R, Xing G, et al. A Python library for probabilistic analysis of single-cell omics
  data. *Nat Biotechnol* 2022;40(2):163-166. doi:10.1038/s41587-021-01206-w. PMID 35132262.
- Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities.
  *Sci Rep* 2019;9(1):5233. doi:10.1038/s41598-019-41695-z. PMID 30914743.
- McInnes L, Healy J, Saul N, Großberger L. UMAP: Uniform Manifold Approximation and Projection.
  *J Open Source Softw* 2018;3(29):861. doi:10.21105/joss.00861.
- Luecken MD, Büttner M, Chaichoompu K, et al. Benchmarking atlas-level data integration in
  single-cell genomics. *Nat Methods* 2022;19(1):41-50. doi:10.1038/s41592-021-01336-8. PMID 34949812.
- Rousseeuw PJ. Silhouettes: A graphical aid to the interpretation and validation of cluster
  analysis. *J Comput Appl Math* 1987;20:53-65. doi:10.1016/0377-0427(87)90125-7.
- Heumos L, Schaar AC, Lance C, et al. Best practices for single-cell analysis across modalities.
  *Nat Rev Genet* 2023;24(8):550-572. doi:10.1038/s41576-023-00586-w. PMID 37002403.
- Single-cell best practices (online book), https://www.sc-best-practices.org: chapters *Feature
  selection* (Schaar A, Kim SH, Heumos L) and *Data integration* (Zappia L, Lücken M, Kim SH,
  Heumos L); the site asks to be cited as Heumos et al. 2023 (above). Read on 2026-10-31.
- Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning
  Library. arXiv 2019. doi:10.48550/arXiv.1912.01703.
- Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. arXiv
  2012. doi:10.48550/arXiv.1201.0490.
- python-igraph 1.0.0: https://github.com/igraph/python-igraph (GPL-2.0-or-later; no DOI).
- harmonypy 2.1.0: https://github.com/slowkow/harmonypy (GPL-3.0-or-later; no DOI).
- Demo data: Arendt-Tranholm A, Sankaranarayanan I, Price TJ. Single nucleus sequencing of human C2
  dorsal root ganglia tissue from fusion surgeries (version 1). SPARC Portal / Pennsieve Discover
  dataset 475, 2025. doi:10.26275/cxuf-ftav. CC BY 4.0. Described by: Arendt-Tranholm A,
  Sankaranarayanan I, Payne C, et al. Transcriptome of the human C2 dorsal root ganglia in C1–2
  arthrodesis surgery: insight for neck pain. *Brain* 2026;149(4):1349-1364.
  doi:10.1093/brain/awaf370. PMID 41032656.
