"""
Finding the inputs through safe_inputs.InputDir (the platform's symlink rule).

Linked layout (placement: subfolder): INPUT_DIR/cells/<sample_id>.h5ad. Without links every
.h5ad under INPUT_DIR is a sample.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from processor.safe_inputs import InputDir

SAMPLE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class Sample:
    sample_id: str
    path: Path


@dataclass
class Inputs:
    root: InputDir
    samples: list[Sample]
    seen: list[str]                # relative paths of every .h5ad accepted by the walk
    skipped: dict[str, str]        # relative path -> reason


def discover(input_dir: Path) -> Inputs:
    root = InputDir(input_dir)
    port = "cells/" if (input_dir / "cells").is_dir() else ""
    skipped: dict[str, str] = {}
    seen: list[str] = []
    samples: dict[str, Sample] = {}
    for p in root.walk():
        rel = root.rel(p)
        if p.suffix.lower() != ".h5ad" or not rel.startswith(port):
            continue
        seen.append(rel)
        if not SAMPLE_ID.match(p.stem):
            skipped[rel] = "file name is not <sample_id>.h5ad with sample_id in [A-Za-z0-9_-]"
        elif p.stem in samples:
            skipped[rel] = f"a second file for sample {p.stem}"
        else:
            samples[p.stem] = Sample(p.stem, p)
    return Inputs(root, sorted(samples.values(), key=lambda s: s.sample_id), seen, skipped)
