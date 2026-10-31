# Third-party software

This app's own code is Apache-2.0 (see `LICENSE`). The image it builds contains the following
third-party components, installed from PyPI at the pinned versions in `requirements.txt` (and
`requirements-cpu.txt` / `requirements-cuda.txt`); their licences apply to the image.

**GPL.** The image bundles and imports **harmonypy** (GPL-3.0-or-later; LICENSE at v2.1.0 is the
GNU GPL version 3) and **python-igraph / igraph** (GPL-2.0-or-later; LICENSE at 1.0.0 is the GNU
GPL version 2, "or any later version"). The image as a whole is therefore distributed under the
GPL-3.0. Our source code stays Apache-2.0. Everything else is under permissive licences
(Apache-2.0, BSD, MIT, PSF, MPL-2.0; MPL-2.0 is file-level copyleft and leaves our code untouched).

Two images are built (`TORCH_FLAVOUR`): the **cpu** variant has the CPU-only PyTorch wheel
(`2.14.1+cpu`) and no NVIDIA software; in the **gpu** variant, the NVIDIA CUDA libraries that come
with the CUDA build of PyTorch are proprietary but redistributable under the NVIDIA software
licence terms shipped in each wheel (the same arrangement as every CUDA PyTorch image).

| Component | What | Licence | Source |
|---|---|---|---|
| scanpy 1.12.5 | `highly_variable_genes(flavor="seurat_v3")`, `pca`, `neighbors`, `leiden`, `umap`. | BSD-3-Clause | https://github.com/scverse/scanpy |
| anndata 0.13.4, mudata 0.4.1 | Reads and writes the `.h5ad` files; data containers of scvi-tools. | BSD-3-Clause | https://github.com/scverse |
| scikit-misc 0.5.3 | LOESS fit inside the seurat_v3 HVG flavour. | BSD-3-Clause | https://github.com/has2k1/scikit-misc |
| harmonypy 2.1.0 | Harmony integration of the PCs (`METHOD=harmony`). | **GPL-3.0-or-later** | https://github.com/slowkow/harmonypy |
| scvi-tools 1.5.2 | scVI model (`METHOD=scvi`). | BSD-3-Clause | https://github.com/scverse/scvi-tools |
| PyTorch 2.14.1 (cpu variant: `+cpu` build; gpu variant: `+cu126` build with triton 3.8.0) | Tensor computation for scVI on CPU or GPU. | BSD-3-Clause style (torch's licence bundle: Apache-2.0, BSD-2/3-Clause, BSL-1.0, ISC, MIT, Zlib); triton MIT | https://pytorch.org |
| nvidia-*-cu12 (cublas, cuda-cupti, cuda-nvrtc, cuda-runtime, cudnn, cufft, cufile, curand, cusolver, cusparse, cusparselt, nccl, nvjitlink, nvshmem, nvtx; versions in `requirements-cuda.txt`); cuda-toolkit 12.6.3, cuda-bindings 12.9.10, cuda-pathfinder 1.8.5 | gpu variant only: CUDA runtime libraries. | NVIDIA proprietary, redistributable (licence in each wheel); cuda-bindings/pathfinder Apache-2.0 | https://developer.nvidia.com/cuda-toolkit |
| Lightning 2.6.6 (lightning, pytorch-lightning, lightning-utilities 0.15.3, torchmetrics 1.9.0) | Training loop used by scvi-tools. | Apache-2.0 | https://github.com/Lightning-AI/pytorch-lightning |
| Pyro 1.9.2 (pyro-ppl, pyro-api 0.1.2) | Dependency of scvi-tools (not used by scVI itself). | Apache-2.0 | https://github.com/pyro-ppl/pyro |
| python-igraph 1.0.0 (igraph, texttable 1.7.2) | Leiden implementation behind `scanpy.tl.leiden(flavor="igraph")`. | **GPL-2.0-or-later**; texttable MIT | https://github.com/igraph/python-igraph |
| umap-learn 0.5.12, pynndescent 0.7.0 | UMAP embedding; approximate nearest neighbours. | BSD-3-Clause; BSD-2-Clause | https://github.com/lmcinnes |
| scikit-learn 1.9.1 | Silhouette and nearest neighbours for the metrics. | BSD-3-Clause | https://scikit-learn.org |
| NumPy 2.5.4, SciPy 1.18.1 | Arrays, sparse matrices. | BSD-3-Clause (NumPy also bundles 0BSD/MIT/Zlib/CC0 parts) | https://numpy.org, https://scipy.org |
| pandas 3.1.0, h5py 3.16.0 | Tables; HDF5 backend of anndata. | BSD-3-Clause | https://pandas.pydata.org, https://www.h5py.org |
| Matplotlib 3.11.2, Pillow 13.0.0 | PNG figures. | Matplotlib licence (PSF-based, BSD-compatible); MIT-CMU | https://matplotlib.org, https://python-pillow.org |
| xarray, sparse, ml_collections, docrep, rich, tensorboard (+ tensorboard-data-server, absl-py, grpcio, protobuf, Markdown, Werkzeug), pydantic (+ pydantic-core, pydantic-settings, annotated-types, typing-inspection, python-dotenv), formulaic, interface_meta, narwhals, msgspec, opt_einsum, cloudpickle, fsspec, aiohttp (+ aiosignal, aiohappyeyeballs, frozenlist, multidict, propcache, yarl, attrs, idna), Jinja2, MarkupSafe, sympy, mpmath, networkx, filelock, setuptools, certifi | Dependencies of scvi-tools, torch and Lightning (versions in `requirements.txt`). | Apache-2.0 / BSD / MIT / MPL-2.0 (certifi) | PyPI |
| numba 0.68.1, llvmlite 0.50.0, statsmodels 0.15.0, patsy 1.0.3, seaborn 0.13.2, joblib, threadpoolctl, natsort, fast-array-utils, legacy-api-wrap, session-info2, scverse-misc, zarr, numcodecs, donfig, google-crc32c, array-api-compat | Dependencies of scanpy / anndata. | BSD-2/3-Clause, MIT, Apache-2.0, MPL-2.0 (fast-array-utils, legacy-api-wrap, session-info2) | PyPI |
| contourpy, cycler, fonttools, kiwisolver, packaging, pyparsing, python-dateutil, six, tqdm, typing_extensions, tzdata, wrapt, PyYAML, Pygments, markdown-it-py, mdurl | Dependencies of the above. | MIT / BSD / Apache-2.0 / MPL-2.0+MIT (tqdm) / PSF-2.0 | PyPI |
| python:3.12.16-slim-bookworm (index digest in the `Dockerfile`) | Base image (Debian 12 + CPython 3.12); docrep's wheel is built from its sdist in a separate build stage. | PSF (Python); Debian packages under their own free licences (the base layer contains GPL system tools) | https://hub.docker.com/_/python |

No models, reference data or other files are downloaded at build or run time.

## Not bundled

- **Demo data** (not in the image or the repository): Pennsieve Discover dataset 475 v1, *Single
  nucleus sequencing of human C2 dorsal root ganglia tissue from fusion surgeries*
  (Arendt-Tranholm A, Sankaranarayanan I, Price TJ; doi:10.26275/cxuf-ftav), CC BY 4.0.
