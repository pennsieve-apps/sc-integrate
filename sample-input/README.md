# sample-input

This app reads the output of `sc-qc` (port `cells`: `<sample_id>.h5ad` per sample with
log-normalized `X`, raw counts in `layers["counts"]`, `obs["sample_id"]` and the sample metadata
columns such as `condition`). The contract is in `workflow-single-cell/docs/conventions.md`.

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/{cells,labels,integration}/
```

Three samples (`s1`, `s2` condition `control`, `s3` condition `pain`) of 400 nuclei and 300 genes.
Five cell types (Neuron, Satellite glia, Schwann cell, Fibroblast, Endothelial) each have 12
marker genes; every sample has its own gene-wise batch factor (log-normal, sigma 0.4) and
sequencing depth (1x, 0.5x, 1.5x). `s3` lacks 6 genes, so the joint analysis uses the 294
shared genes. The truth is in `obs["annotation"]` and `truth.csv`.

## Real data

Run `sc-prep` and `sc-qc` on a 10x dataset, for example the workflow's demo data
(`scripts/fetch-sample-data.sh` in the workflow repository): Pennsieve Discover dataset 475,
*Single nucleus sequencing of human C2 dorsal root ganglia tissue from fusion surgeries*
(Arendt-Tranholm A, Sankaranarayanan I, Price TJ; doi:10.26275/cxuf-ftav; CC BY 4.0). Then:

```sh
docker run --rm --platform linux/amd64 --memory 4g --network none --read-only --tmpfs /tmp \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output \
  -v /path/to/sc-qc/output/cells:/data/input/cells:ro -v "$PWD/data/output:/data/output" sc-integrate:dev
```
