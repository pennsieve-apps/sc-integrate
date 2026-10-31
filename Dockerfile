FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /processor

# PyTorch build: cpu or cuda. app.yml's variants set it (cpu variant: CPU-only torch from the PyTorch
# CPU index, no NVIDIA libraries; gpu variant: the CUDA 13.2 build, which also runs on CPU). The
# default is cpu for a builder that passes no build arguments.
ARG TORCH_FLAVOUR=cpu

# The complete pinned dependency set (no resolver: see requirements.txt; wheels, plus the pure-Python
# sdist of docrep).
COPY requirements.txt requirements-${TORCH_FLAVOUR}.txt /processor/
RUN pip install --no-deps --prefer-binary \
        -r /processor/requirements.txt -r /processor/requirements-${TORCH_FLAVOUR}.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -path '*/torch/*' -prune -o -type d -name tests -prune -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/torch/include

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache \
    TORCH_FLAVOUR=${TORCH_FLAVOUR}

# Build matplotlib's font cache and import the stack once, so a run needs no writes outside /tmp and
# no network.
RUN python -c "import matplotlib.pyplot, scanpy, harmonypy, igraph, skmisc.loess, scvi, torch" \
    && mkdir -p /tmp/matplotlib /tmp/numba /tmp/cache && chmod -R a+rwX /tmp/matplotlib /tmp/numba /tmp/cache

# Runs as uid 1000 on the platform; nothing here needs root. PyTorch looks the user up (getpass), so
# uid 1000 gets a passwd entry.
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]
