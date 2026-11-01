"""Keep written numbers in step with generated numbers.

Results are regenerated; text is not. A value quoted in a report, a caption or a
table drifts from the artifact it came from as soon as the pipeline is rerun,
and nothing fails. This module makes the drift visible: you register the values
that are known to be obsolete, point it at the document tree, and it reports
every remaining occurrence with a file and line.

It also flags artifacts that share a name but not a content hash, which is the
other half of the same failure mode: two logs from different runs sitting side
by side under the same filename.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

__all__ = ["StaleHit", "DuplicateArtifact", "scan_stale", "find_duplicate_artifacts"]

DEFAULT_INCLUDE = (".md", ".tex", ".txt", ".rst")
DEFAULT_EXCLUDE = (
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".pytest_cache",
    "build",
    "dist",
)


@dataclass
class StaleHit:
    path: str
    line: int
    text: str
    value: str
    label: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return "%s:%d  [%s] %s" % (self.path, self.line, self.label, self.value)


@dataclass
class DuplicateArtifact:
    name: str
    paths: List[str]
    hashes: List[str]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return "%s: %d copies with %d distinct contents" % (
            self.name,
            len(self.paths),
            len(set(self.hashes)),
        )


def _walk(root: str, include: Sequence[str], exclude: Sequence[str]):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fn in filenames:
            if fn.lower().endswith(tuple(include)):
                yield os.path.join(dirpath, fn)


def scan_stale(
    root: str,
    stale: Dict[str, str],
    include: Sequence[str] = DEFAULT_INCLUDE,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    regex: bool = False,
    max_hits: int = 1000,
) -> List[StaleHit]:
    """Find obsolete values still present in the document tree.

    Parameters
    ----------
    root:
        Directory to scan.
    stale:
        Mapping of the text to search for to a human label describing what it
        used to mean, e.g. ``{"57.93": "drying time before the grid fix"}``.
    regex:
        Treat the keys as regular expressions. Off by default, because numeric
        literals are awkward to escape and the plain-substring case is the
        common one.

    Returns one :class:`StaleHit` per occurrence, ordered by path then line.
    """
    if not stale:
        return []

    if regex:
        patterns = [(re.compile(k), k, label) for k, label in stale.items()]
    else:
        patterns = [(None, k, label) for k, label in stale.items()]

    hits: List[StaleHit] = []
    for path in sorted(_walk(root, include, exclude)):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue

        for lineno, line in enumerate(lines, 1):
            for compiled, key, label in patterns:
                found = compiled.search(line) if compiled is not None else (key in line)
                if found:
                    hits.append(
                        StaleHit(
                            path=os.path.relpath(path, root),
                            line=lineno,
                            text=line.strip()[:200],
                            value=key,
                            label=label,
                        )
                    )
                    if len(hits) >= max_hits:
                        return hits
    return hits


def find_duplicate_artifacts(
    roots: Iterable[str],
    suffixes: Sequence[str] = (".json", ".txt", ".log", ".csv"),
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
) -> List[DuplicateArtifact]:
    """Group files by basename and keep the groups whose contents differ.

    Two logs with the same name but different content is a reproduction hazard:
    the reader has no way to tell which run produced which. Only groups with
    more than one distinct hash are returned.
    """
    groups: Dict[str, List[str]] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude]
            for fn in filenames:
                if fn.lower().endswith(tuple(suffixes)):
                    groups.setdefault(fn, []).append(os.path.join(dirpath, fn))

    out: List[DuplicateArtifact] = []
    for name, paths in sorted(groups.items()):
        if len(paths) < 2:
            continue
        hashes = []
        for p in paths:
            try:
                with open(p, "rb") as fh:
                    hashes.append(hashlib.md5(fh.read()).hexdigest()[:10])
            except OSError:
                hashes.append("unreadable")
        if len(set(hashes)) > 1:
            out.append(DuplicateArtifact(name=name, paths=paths, hashes=hashes))
    return out
