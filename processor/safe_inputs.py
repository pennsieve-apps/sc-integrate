"""
Safe access to a processor's $INPUT_DIR, shared verbatim by every app (processor/safe_inputs.py).
The canonical copy is workflow-single-cell/shared/safe_inputs.py; scripts/check-shared.sh
checks the vendored copies. Do not edit a vendored copy.

The platform's links are the entries of INPUT_DIR and of a REAL (non-symlink) INPUT_DIR/<port>/
folder. A file is allowed when it resolves inside INPUT_DIR or inside the target of one of those
links. A link's target is taken one hop (its own link text, parents resolved): when the target is
itself a link (an upstream app wrote a symlink into its output), what that one points at is not
trusted. Deeper links that leave are skipped, each logged once by relative path.

`InputDir.walk()` lists the accepted files. Any path an app builds itself (joining, globbing) must
go through `inside()` / `check()` / `open_checked()` or a capped reader, which check again.

Standard library only; pandas, openpyxl and h5py are imported only by the helpers that need them.
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import math
import os
import stat
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

log = logging.getLogger("safe_inputs")

CGROUP = Path("/sys/fs/cgroup")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class UnsafeInput(ValueError):
    """A path or file this module refuses: outside the input, not a regular file, too large, unsafe content."""


def _link_target(link: Path) -> str:
    """Where a link points, one hop: its parent folders resolved, its own target not followed further."""
    t = os.path.join(os.path.realpath(link.parent), os.readlink(link)).rstrip(os.sep) or os.sep
    head, name = os.path.split(t)
    if name in ("", ".", ".."):
        return os.path.realpath(t)
    return os.path.join(os.path.realpath(head), name)


def _under(real: str, root: str) -> bool:
    return real == root or real.startswith(root if root.endswith(os.sep) else root + os.sep)


class InputDir:
    """The input folder of one run. `roots` are the real paths files may resolve to."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.roots = [os.path.realpath(self.root)]
        self._logged: set[str] = set()
        for top in sorted(self.root.iterdir()):
            if top.is_symlink():
                self.roots.append(_link_target(top))
            elif top.is_dir():                       # a real port folder: its links are the platform's
                self.roots += [_link_target(e) for e in sorted(top.iterdir()) if e.is_symlink()]

    def rel(self, path: str | os.PathLike) -> str:
        """The path relative to INPUT_DIR for logs and provenance (as given when it is not under it)."""
        try:
            return Path(path).relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def _skip(self, path: str | os.PathLike, reason: str) -> None:
        r = self.rel(path)
        if r not in self._logged:
            self._logged.add(r)
            log.warning("skipped %s: %s", r, reason)

    def _ok(self, real: str) -> bool:
        return any(_under(real, r) for r in self.roots)

    def walk(self) -> list[Path]:
        """Every regular file under INPUT_DIR that resolves inside the allowed roots, in sorted walk order.
        Symlinks are followed; hidden entries are ignored; a folder is visited once (no loops)."""
        found: list[Path] = []
        seen = {os.path.realpath(self.root)}
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=True):
            keep = []
            for d in sorted(n for n in dirnames if not n.startswith(".")):
                p = os.path.join(dirpath, d)
                real = os.path.realpath(p)
                if not self._ok(real):
                    self._skip(p, "resolves outside the input")
                elif real in seen:
                    self._skip(p, "a folder already walked (link loop or second link)")
                else:
                    seen.add(real)
                    keep.append(d)
            dirnames[:] = keep
            for f in sorted(n for n in filenames if not n.startswith(".")):
                p = Path(dirpath, f)
                real = os.path.realpath(p)
                if not os.path.isfile(real):
                    self._skip(p, "not a regular file (broken link, link loop or special file)")
                elif not self._ok(real):
                    self._skip(p, "resolves outside the input")
                else:
                    found.append(p)
        return found

    def inside(self, path: str | os.PathLike) -> bool:
        """Whether the path (relative ones are taken under INPUT_DIR) resolves inside the allowed roots."""
        p = Path(path)
        return self._ok(os.path.realpath(p if p.is_absolute() else self.root / p))

    def check(self, path: str | os.PathLike) -> str:
        """The resolved path of an allowed regular file; raises UnsafeInput otherwise."""
        p = Path(path)
        real = os.path.realpath(p if p.is_absolute() else self.root / p)
        if not self._ok(real):
            self._skip(path, "resolves outside the input")
            raise UnsafeInput(f"{self.rel(path)} resolves outside the input")
        try:
            if not stat.S_ISREG(os.stat(real).st_mode):
                raise UnsafeInput(f"{self.rel(path)} is not a regular file")
        except OSError as exc:
            raise UnsafeInput(f"{self.rel(path)} cannot be read: {exc.strerror}") from None
        return real

    def open_checked(self, path: str | os.PathLike) -> BinaryIO:
        """The file opened for binary reading, after check()."""
        return open(self.check(path), "rb")

    def read_bytes(self, path: str | os.PathLike, max_bytes: int) -> bytes:
        """The file's bytes; UnsafeInput when it is larger than max_bytes."""
        with self.open_checked(path) as fh:
            too_large = os.fstat(fh.fileno()).st_size > max_bytes
            data = b"" if too_large else fh.read(max_bytes + 1)     # the file may grow while read
        if too_large or len(data) > max_bytes:
            raise UnsafeInput(f"{self.rel(path)} is larger than {max_bytes} bytes")
        return data

    def read_text(self, path: str | os.PathLike, max_bytes: int, encoding: str = "utf-8-sig") -> str:
        """The file as text (a byte-order mark dropped); UnicodeDecodeError (a ValueError) on bad bytes."""
        return self.read_bytes(path, max_bytes).decode(encoding)

    def read_json(self, path: str | os.PathLike, max_bytes: int) -> Any:
        """The parsed JSON value; ValueError when it is not JSON (or nested too deeply)."""
        try:
            return json.loads(self.read_text(path, max_bytes))
        except RecursionError:
            raise UnsafeInput(f"{self.rel(path)} is nested too deeply") from None

    def read_csv(self, path: str | os.PathLike, max_bytes: int, **kwargs):
        """pandas.read_csv of the file. A .gz file is decompressed here, capped at max_bytes uncompressed;
        any other file is read as is (the `compression` argument is overridden: no other decompression)."""
        import pandas as pd

        kwargs["compression"] = None
        with self.open_checked(path) as fh:
            if os.fstat(fh.fileno()).st_size > max_bytes:
                raise UnsafeInput(f"{self.rel(path)} is larger than {max_bytes} bytes")
            if not str(path).lower().endswith(".gz"):
                return pd.read_csv(fh, **kwargs)
            with gzip.GzipFile(fileobj=fh) as gz:
                data = gz.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise UnsafeInput(f"{self.rel(path)} is larger than {max_bytes} bytes uncompressed")
        return pd.read_csv(io.BytesIO(data), **kwargs)

    def xlsx_uncompressed_size(self, path: str | os.PathLike) -> int:
        """The sum of the declared sizes of an .xlsx (zip) file's members: compare it with a cap before
        openpyxl opens the file (Python's zipfile never inflates a member beyond its declared size)."""
        with self.open_checked(path) as fh:
            try:
                with zipfile.ZipFile(fh) as z:
                    return sum(i.file_size for i in z.infolist())
            except (zipfile.BadZipFile, EOFError) as exc:
                raise UnsafeInput(f"{self.rel(path)} is not a valid xlsx/zip file: {exc}") from None

    def check_h5(self, path: str | os.PathLike) -> None:
        """Raises UnsafeInput when an HDF5 file reaches other files: an external link, a dataset with
        external storage, or a virtual dataset. Soft links are paths within the same file (HDF5 resolves
        them there), so they cannot leave it except through an external link, which is refused."""
        import h5py

        real = self.check(path)

        def visit(group, seen: set) -> None:
            for name in group:
                link = group.get(name, getlink=True)
                where = f"{self.rel(path)}:{group.name.rstrip('/')}/{name}"
                if isinstance(link, h5py.ExternalLink):
                    raise UnsafeInput(f"{where} is an external link")
                if isinstance(link, h5py.SoftLink):
                    continue
                obj = group[name]
                if obj.id in seen:
                    continue
                seen.add(obj.id)
                if isinstance(obj, h5py.Dataset):
                    if obj.external:
                        raise UnsafeInput(f"{where} stores its data in external files")
                    if obj.is_virtual:
                        raise UnsafeInput(f"{where} is a virtual dataset")
                elif isinstance(obj, h5py.Group):
                    visit(obj, seen)

        try:
            with h5py.File(real, "r", locking=False) as f:
                visit(f, set())
        except OSError as exc:
            raise UnsafeInput(f"{self.rel(path)} is not a readable HDF5 file: {exc}") from None

    def inline_png(self, path: str | os.PathLike, max_bytes: int) -> str | None:
        """A data: URI for an allowed PNG file no larger than max_bytes; None (logged) otherwise."""
        try:
            data = self.read_bytes(path, max_bytes)
        except (UnsafeInput, OSError) as exc:
            self._skip(path, f"not inlined: {exc}")
            return None
        if not data.startswith(PNG_MAGIC):
            self._skip(path, "not inlined: not a PNG file")
            return None
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def assert_defusedxml() -> None:
    """Raises RuntimeError unless openpyxl parses with defusedxml (no entity expansion)."""
    import openpyxl

    if not getattr(openpyxl, "DEFUSEDXML", False):
        raise RuntimeError("openpyxl is not using defusedxml; install defusedxml before reading .xlsx files")


def _cgroup_cpus() -> int | None:
    """CPUs from the cgroup v2 cpu.max (or v1 CFS quota), rounded up; None without a quota."""
    try:
        quota, period = (CGROUP / "cpu.max").read_text().split()[:2]
        return None if quota == "max" else max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int((CGROUP / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((CGROUP / "cpu" / "cpu.cfs_period_us").read_text())
        return max(1, math.ceil(quota / period)) if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


def cpu_threads(requested: int | str | None = 0) -> int:
    """Threads to use: `requested` when > 0 (THREADS), else the cgroup CPU quota (no more than the CPUs
    this process may run on), else the CPUs this process may run on."""
    n = int(requested or 0)
    if n > 0:
        return n
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:                       # macOS
        cpus = os.cpu_count() or 1
    quota = _cgroup_cpus()
    return min(quota, cpus) if quota else cpus
