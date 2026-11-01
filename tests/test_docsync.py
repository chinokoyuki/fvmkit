import os

from fvmkit.docsync import find_duplicate_artifacts, scan_stale


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_scan_stale_finds_known_values(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "paper.md"), "line one\nThe drying time is 57.93 h.\n")
    _write(os.path.join(root, "notes", "draft.tex"), "value: 57.93\n")
    _write(os.path.join(root, "ok.md"), "nothing here\n")

    hits = scan_stale(root, {"57.93": "pre-fix drying time"})
    assert len(hits) == 2
    assert {h.path.replace("\\", "/") for h in hits} == {"paper.md", "notes/draft.tex"}
    assert all(h.label == "pre-fix drying time" for h in hits)


def test_scan_stale_respects_extension_filter(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "a.py"), "x = 57.93\n")
    hits = scan_stale(root, {"57.93": "old"}, include=(".md",))
    assert hits == []


def test_scan_stale_supports_regex(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "a.md"), "values 57.93 and 57.95\n")
    hits = scan_stale(root, {r"57\.9\d": "near-miss"}, regex=True)
    assert len(hits) == 1


def test_scan_stale_skips_excluded_directories(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, ".git", "x.md"), "57.93\n")
    _write(os.path.join(root, "keep.md"), "57.93\n")
    hits = scan_stale(root, {"57.93": "old"})
    assert len(hits) == 1
    assert hits[0].path == "keep.md"


def test_find_duplicate_artifacts_flags_conflicting_copies(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "run_a", "solve_log.txt"), "N=160\n")
    _write(os.path.join(root, "run_b", "solve_log.txt"), "N=320\n")
    _write(os.path.join(root, "run_a", "other.txt"), "same\n")
    _write(os.path.join(root, "run_b", "other.txt"), "same\n")

    dups = find_duplicate_artifacts([root])
    names = [d.name for d in dups]
    assert names == ["solve_log.txt"]
    assert len(set(dups[0].hashes)) == 2


def test_find_duplicate_artifacts_returns_nothing_for_identical_copies(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "a", "log.txt"), "same\n")
    _write(os.path.join(root, "b", "log.txt"), "same\n")
    assert find_duplicate_artifacts([root]) == []
