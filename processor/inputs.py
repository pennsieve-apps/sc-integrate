"""
Finding the inputs.

Linked layout (placement: subfolder): INPUT_DIR/cells/<sample_id>.h5ad. Without links the
same files are found anywhere under INPUT_DIR. Symlinks are followed: the platform's own
links (the entries of INPUT_DIR and of INPUT_DIR/<port>/) may point anywhere, but a file or
folder further down whose real path leaves INPUT_DIR and those link targets is skipped.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

SAMPLE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class Sample:
    sample_id: str
    path: Path


@dataclass
class Inputs:
    samples: list[Sample]
    seen: list[Path]
    skipped: dict[str, str]        # file -> reason


def allowed_roots(input_dir: Path) -> list[Path]:
    """INPUT_DIR and the real paths of the platform's links: the entries of INPUT_DIR and of its
    port folders (INPUT_DIR/<port>/)."""
    roots = [input_dir.resolve()]
    for entry in input_dir.iterdir():
        roots.append(entry.resolve())
        if entry.is_dir():
            roots.extend(e.resolve() for e in entry.iterdir())
    return roots


def inside(path: Path, roots: list[Path]) -> bool:
    real = path.resolve()
    return any(real == r or real.is_relative_to(r) for r in roots)


def walk(root: Path, roots: list[Path]) -> list[Path]:
    """Every file under root, following symlinks that stay inside `roots`, each real folder once."""
    found: list[Path] = []
    visited: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = Path(dirpath).resolve()
        if real in visited or not inside(Path(dirpath), roots):
            dirnames[:] = []
            continue
        visited.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        found.extend(Path(dirpath) / f for f in sorted(filenames) if not f.startswith("."))
    return found


def discover(input_dir: Path) -> Inputs:
    roots = allowed_roots(input_dir)
    folder = input_dir / "cells"
    base = folder if folder.is_dir() else input_dir
    skipped: dict[str, str] = {}
    seen: list[Path] = []
    samples: dict[str, Sample] = {}
    for p in walk(base, roots):
        if p.suffix.lower() != ".h5ad":
            continue
        rel = str(p.relative_to(input_dir))
        seen.append(p)
        if not inside(p, roots):
            skipped[rel] = "links outside INPUT_DIR"
        elif not SAMPLE_ID.match(p.stem):
            skipped[rel] = "file name is not <sample_id>.h5ad with sample_id in [A-Za-z0-9_-]"
        elif p.stem in samples:
            skipped[rel] = f"a second file for sample {p.stem}"
        else:
            samples[p.stem] = Sample(p.stem, p)
    return Inputs(sorted(samples.values(), key=lambda s: s.sample_id), seen, skipped)
