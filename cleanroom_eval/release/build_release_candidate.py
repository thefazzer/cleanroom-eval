"""Build and audit the isolated clean-room release candidate (#145 Phase D).

Produces a deterministic archive from a STRICT ALLOWLIST — never from the
repository at large — and runs the release-audit gates against the exact
archive bytes. The repository's history is contaminated and is never
exported; a publishable candidate goes to a fresh repository with new
history.

Usage:
    python -m cleanroom_eval.release.build_release_candidate [--out DIR]

Outputs (under --out, default ~/.local/share/finexhaust/cleanroom-release):
    cleanroom-eval-release.tar.gz    the candidate archive
    release-audit.json               gate results + terminal state

The audit report alone is committed back to the repository; the archive
stays local until the owner's publication decision (Phase E).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_NAME = "cleanroom-eval-release.tar.gz"
FIXED_MTIME = 1787788800  # 2026-08-23T00:00:00Z — deterministic archives

# ---- allowlist ------------------------------------------------------------
ALLOW_DIRS = (
    ("cleanroom_eval", "*.py"),
    ("cleanroom_eval/schemas", "**/*"),
    ("cleanroom_eval/assets", "**/*"),
    ("cleanroom_eval/mock_training", "**/*"),
    ("cleanroom_eval/evidence/gates-2026-08", "*"),
    ("cleanroom_eval/evidence", "bind_gates_evidence.py"),
    ("cleanroom_eval/release", "build_release_candidate.py"),
)
ALLOW_FILES = (
    "cleanroom_eval/README.md",
    "cleanroom_corpus/eval_adapters.py",
    "cleanroom_corpus/eval_export.py",
    "requirements-test.txt",
    "LICENSE",
)
# Evaluator-only material must never ship, allowlist notwithstanding.
DENY_NAME_PATTERNS = (
    re.compile(r"oracle", re.IGNORECASE),
    re.compile(r"\.pyc$"),
    re.compile(r"__pycache__"),
)
MINIMAL_CORPUS_INIT = (
    '"""Release packaging: adapter surface only.\n\n'
    "The full synthetic-corpus generator is not part of the clean-room\n"
    'release; only the evaluation adapters and export contract ship."""\n'
)
SECURITY_STATEMENT = """# Security and release statement

This archive contains ONLY the clean-room evaluation environment: code,
schemas, fictitious CLEANROOM_SYNTHETIC episodes, task contracts, the
content-safe evidence summary, and this statement. It is built from an
explicit allowlist — never from the source repository's history.

- Every episode, name, institution and identifier is invented. The evidence
  summary binds results by hash and contains no transcripts, credentials or
  source-derived records.
- Episodes deliberately embed canary values and reward traps; they are part
  of the published task boundary. A model trained on this release loses
  canary-based contamination detection — the standard public-benchmark
  caveat.
- The evaluator-only citation oracle and all private gold are excluded and
  their absence is enforced by the release audit's leakage gate. The README
  step `verify_citation_task_set()` recomputes oracle invariants and is
  therefore evaluator-only: it is not runnable from this archive. Every
  other documented step (verify-bundle, verify-set, the preregistered
  scripted baseline) runs from the archive with `pip install -r
  requirements.txt` and no network access at run time.
- Report issues to the repository owner; do not open public issues that
  quote unpublished material.
"""
RUNTIME_REQUIREMENTS = "cryptography==46.0.5\njsonschema==4.26.0\n"


def _iter_allowlist() -> list[tuple[Path, str]]:
    """Return (source_path, archive_relpath) pairs, deny-filtered."""
    seen: dict[str, Path] = {}
    for base, pattern in ALLOW_DIRS:
        root = REPO_ROOT / base
        if not root.is_dir():
            continue
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            rel = str(path.relative_to(REPO_ROOT))
            if any(p.search(rel) for p in DENY_NAME_PATTERNS):
                continue
            seen[rel] = path
    for rel in ALLOW_FILES:
        path = REPO_ROOT / rel
        if path.is_file() and not any(p.search(rel) for p in DENY_NAME_PATTERNS):
            seen[rel] = path
    return [(seen[rel], rel) for rel in sorted(seen)]


def build_archive(out_dir: Path) -> Path:
    entries = _iter_allowlist()
    manifest_lines = []
    generated = {
        "cleanroom_corpus/__init__.py": MINIMAL_CORPUS_INIT.encode("utf-8"),
        "SECURITY-RELEASE.md": SECURITY_STATEMENT.encode("utf-8"),
        "requirements.txt": RUNTIME_REQUIREMENTS.encode("utf-8"),
    }
    payloads: list[tuple[str, bytes]] = []
    for path, rel in entries:
        payloads.append((rel, path.read_bytes()))
    for rel, blob in generated.items():
        payloads.append((rel, blob))
    payloads.sort(key=lambda pair: pair[0])
    for rel, blob in payloads:
        manifest_lines.append(f"{hashlib.sha256(blob).hexdigest()}  {rel}")
    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    payloads.append(("RELEASE-MANIFEST.sha256", manifest))

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / ARCHIVE_NAME
    # gzip with mtime=0 for byte-determinism
    import gzip

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for rel, blob in payloads:
            info = tarfile.TarInfo(name=rel)
            info.size = len(blob)
            info.mtime = FIXED_MTIME
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(blob))
    archive_path.write_bytes(gzip.compress(raw.getvalue(), mtime=0))
    return archive_path


# ---- gates ---------------------------------------------------------------
SECRET_PATTERNS = (
    r"AKIA[0-9A-Z]{16}", r"sk-[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{36}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}", r"cfat_[A-Za-z0-9]{20,}",
    r"-----BEGIN (?:RSA|OPENSSH|EC|PGP) PRIVATE KEY-----", r"AIza[0-9A-Za-z_-]{35}",
    r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]{16,}",
)
INTERNAL_PATTERNS = (
    r"/home/[a-z]+/", r"/mnt/[ce]/", r"/Users/[A-Za-z]+/",
    r"(?i)\bpfarrington\b", r"(?i)C:\\\\Users\\\\",
)


def _archive_members(archive_path: Path) -> list[tuple[tarfile.TarInfo, bytes]]:
    members = []
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for info in tar.getmembers():
            blob = tar.extractfile(info).read() if info.isfile() else b""
            members.append((info, blob))
    return members


def _text(blob: bytes) -> str:
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def gate_secret_scan(members) -> dict:
    hits = []
    for info, blob in members:
        text = _text(blob)
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text):
                hits.append({"file": info.name, "pattern": pattern})
    return {"status": "PASS" if not hits else "FAIL", "hits": hits}


def gate_internal_identifier_scan(members) -> dict:
    hits = []
    for info, blob in members:
        text = _text(blob)
        for pattern in INTERNAL_PATTERNS:
            for match in set(re.findall(pattern, text)):
                hits.append({"file": info.name, "pattern": pattern, "match": match})
    return {"status": "PASS" if not hits else "FAIL", "hits": hits[:50]}


def _shingles(text: str, width: int = 10):
    words = re.findall(r"[a-z0-9']+", text.lower())
    for index in range(len(words) - width + 1):
        yield " ".join(words[index : index + width])


def gate_source_shingle_overlap(members, corpora_dir: Path) -> dict:
    corpus_files = sorted(corpora_dir.glob("*.txt")) if corpora_dir.is_dir() else []
    if not corpus_files:
        return {"status": "NOT_RUN", "reason": "record corpus not present on this machine"}
    corpus_shingles: set[str] = set()
    for path in corpus_files:
        corpus_shingles.update(_shingles(path.read_text(encoding="utf-8", errors="replace")))
    overlaps = []
    for info, blob in members:
        text = _text(blob)
        if not text:
            continue
        matched = {s for s in _shingles(text) if s in corpus_shingles}
        if matched:
            overlaps.append({"file": info.name, "overlapping_shingles": len(matched),
                             "example_hash": hashlib.sha256(sorted(matched)[0].encode()).hexdigest()[:16]})
    return {
        "status": "PASS" if not overlaps else "FAIL",
        "corpora_scanned": len(corpus_files),
        "shingle_width_words": 10,
        "overlaps": overlaps,
    }


def gate_private_gold_leak(members) -> dict:
    leaks = [
        {"file": info.name, "reason": "matches evaluator-only naming"}
        for info, _ in members
        if re.search(r"oracle|private[-_]gold", info.name, re.IGNORECASE)
    ]
    # evaluator-only oracle CONTENT must not appear under another name
    oracle = REPO_ROOT / "cleanroom_eval/assets/citation_tasks/citation-task-oracle.v1.jsonl"
    if oracle.is_file():
        needles = set()
        for line in oracle.read_text(encoding="utf-8").splitlines()[:200]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            blob = json.dumps(row, sort_keys=True)
            needles.add(blob[:120])
        for info, blob in members:
            text = _text(blob)
            if text and any(needle[:60] in text for needle in needles):
                leaks.append({"file": info.name, "reason": "embeds oracle-derived content"})
    return {"status": "PASS" if not leaks else "FAIL", "leaks": leaks}


def gate_path_metadata_scan(members) -> dict:
    problems = []
    for info, _ in members:
        if info.name.startswith("/") or ".." in Path(info.name).parts:
            problems.append({"entry": info.name, "reason": "unsafe path"})
        if info.issym() or info.islnk():
            problems.append({"entry": info.name, "reason": "link entry"})
        if info.uid or info.gid or info.uname or info.gname:
            problems.append({"entry": info.name, "reason": "identity-bearing ownership metadata"})
    return {"status": "PASS" if not problems else "FAIL", "problems": problems}


def gate_license_inventory(members) -> dict:
    names = {info.name for info, _ in members}
    dependencies = []
    for info, blob in members:
        if info.name == "requirements-test.txt":
            dependencies = [
                line.strip() for line in _text(blob).splitlines()
                if line.strip() and not line.startswith("#")
            ]
    has_license = "LICENSE" in names
    return {
        "status": "PASS" if has_license else "FAIL",
        "license_present": has_license,
        "declared_dependencies": dependencies,
        "note": None if has_license else "no LICENSE file exists in the repository; owner must choose one",
    }


def gate_deterministic_rebuild(archive_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as scratch:
        second = build_archive(Path(scratch))
        first_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        second_sha = hashlib.sha256(second.read_bytes()).hexdigest()
    return {
        "status": "PASS" if first_sha == second_sha else "FAIL",
        "archive_sha256": first_sha,
        "rebuild_sha256": second_sha,
    }


def gate_manifest_verification(members) -> dict:
    by_name = {info.name: blob for info, blob in members}
    manifest = _text(by_name.get("RELEASE-MANIFEST.sha256", b""))
    if not manifest:
        return {"status": "FAIL", "reason": "manifest missing"}
    mismatches = []
    listed = set()
    for line in manifest.splitlines():
        sha, _, rel = line.partition("  ")
        listed.add(rel)
        if rel not in by_name:
            mismatches.append({"file": rel, "reason": "listed but absent"})
        elif hashlib.sha256(by_name[rel]).hexdigest() != sha:
            mismatches.append({"file": rel, "reason": "hash mismatch"})
    for name in by_name:
        if name not in listed and name != "RELEASE-MANIFEST.sha256":
            mismatches.append({"file": name, "reason": "present but unlisted"})
    return {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches}


def gate_import_smoke(archive_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as scratch:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            tar.extractall(scratch, filter="data")
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]);\n"
            "import cleanroom_eval.contract, cleanroom_eval.free_run, cleanroom_eval.fire_gates\n"
            "from cleanroom_eval.contract import ASSET_DIR\n"
            "import cleanroom_eval.episode_contract\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe, scratch],
            capture_output=True, text=True, timeout=120,
            cwd=scratch,
        )
    ok = result.returncode == 0 and "IMPORT_OK" in result.stdout
    return {
        "status": "PASS" if ok else "FAIL",
        "detail": None if ok else (result.stderr.strip().splitlines() or ["no output"])[-1],
    }


def gate_tutorial_replay(archive_path: Path) -> dict:
    """Replay the release-runnable tutorial core in an isolated fresh venv.

    Extracts the archive, creates a clean virtualenv, installs the release's
    own requirements.txt, and runs the documented steps: sealed-set
    verification for both episode sets, preregistration, and the scripted
    baseline (40 episodes, in-process, no network at run time). Machine-level
    isolation is a recorded limitation, not silently claimed.
    """
    steps_run: list[dict] = []
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        with tarfile.open(archive_path, mode="r:gz") as tar:
            tar.extractall(root, filter="data")
        venv_dir = root / "venv"
        bootstrap = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True, text=True, timeout=180,
        )
        if bootstrap.returncode != 0:
            return {"status": "FAIL", "detail": "venv creation failed"}
        python = venv_dir / "bin" / "python"
        install = subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "-r", str(root / "requirements.txt")],
            capture_output=True, text=True, timeout=600,
        )
        if install.returncode != 0:
            return {
                "status": "FAIL",
                "detail": "dependency install failed: " + install.stderr.strip().splitlines()[-1][:200],
            }
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(root),
            "HOME": scratch,
        }
        commands = (
            ["-m", "cleanroom_eval.contract", "verify-bundle"],
            ["-m", "cleanroom_eval.contract", "verify-set"],
            ["-m", "cleanroom_eval.fire_gates", "preregister", "--out", "runs"],
            ["-m", "cleanroom_eval.fire_gates", "baseline", "--out", "runs"],
        )
        for command in commands:
            result = subprocess.run(
                [str(python), *command],
                capture_output=True, text=True, timeout=1200, cwd=root, env=environment,
            )
            steps_run.append({"step": " ".join(command), "returncode": result.returncode})
            if result.returncode != 0:
                detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
                return {
                    "status": "FAIL",
                    "steps": steps_run,
                    "detail": (detail or ["no output"])[-1][:300],
                }
    return {
        "status": "PASS",
        "environment": "isolated-venv",
        "steps": steps_run,
        "limitation": "replayed in a fresh venv on the build machine; separate-machine replay not performed",
    }


def terminal_state(gates: dict) -> str:
    if gates["private_gold_leak"]["status"] == "FAIL":
        return "PRIVATE_GOLD_LEAK"
    if gates["secret_scan"]["status"] == "FAIL" or gates["internal_identifier_scan"]["status"] == "FAIL":
        return "SECRET_OR_IDENTIFIER_LEAK"
    if gates["source_shingle_overlap"]["status"] == "FAIL":
        return "SOURCE_OVERLAP_DETECTED"
    if gates["license_inventory"]["status"] == "FAIL":
        return "RIGHTS_OR_LICENSE_UNRESOLVED"
    if gates["deterministic_rebuild"]["status"] == "FAIL" or gates["import_smoke"]["status"] == "FAIL":
        return "RUNTIME_NOT_REPRODUCIBLE"
    if gates["clean_machine_tutorial"]["status"] != "PASS":
        return "RUNTIME_NOT_REPRODUCIBLE"
    if any(g["status"] != "PASS" or g.get("limitation") for g in gates.values()):
        return "RELEASE_AUDIT_PASS_WITH_LIMITATIONS"
    return "RELEASE_AUDIT_PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path.home() / ".local/share/finexhaust/cleanroom-release"))
    parser.add_argument("--corpora", default=str(Path.home() / ".local/share/finexhaust/corpora"))
    args = parser.parse_args()
    out_dir = Path(args.out)

    archive = build_archive(out_dir)
    members = _archive_members(archive)
    gates = {
        "secret_scan": gate_secret_scan(members),
        "internal_identifier_scan": gate_internal_identifier_scan(members),
        "source_shingle_overlap": gate_source_shingle_overlap(members, Path(args.corpora)),
        "private_gold_leak": gate_private_gold_leak(members),
        "path_metadata_scan": gate_path_metadata_scan(members),
        "license_inventory": gate_license_inventory(members),
        "deterministic_rebuild": gate_deterministic_rebuild(archive),
        "manifest_verification": gate_manifest_verification(members),
        "import_smoke": gate_import_smoke(archive),
        "clean_machine_tutorial": gate_tutorial_replay(archive),
    }
    report = {
        "schema": "cleanroom.release-audit/v1",
        "archive": ARCHIVE_NAME,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "file_count": len(members),
        "archive_bytes": archive.stat().st_size,
        "gates": gates,
        "terminal_state": terminal_state(gates),
    }
    (out_dir / "release-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {name: g["status"] for name, g in gates.items()}
    print(json.dumps({"archive": str(archive), "gates": summary,
                      "terminal_state": report["terminal_state"]}, indent=2))


if __name__ == "__main__":
    main()
