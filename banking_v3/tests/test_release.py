from pathlib import Path

import pytest

from cleanroom_eval import certify
from cleanroom_eval.release import build_release_candidate as build


def test_no_legacy_refusal_certificate_can_be_issued(tmp_path):
    with pytest.raises(certify.CertificationError, match="RefusePolicy"):
        certify.run_certification(prereg_path=tmp_path, sealed_path=tmp_path,
            evidence_path=tmp_path, runs_root=tmp_path, out_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_public_release_allows_v3_runtime_but_not_private_data():
    names = [name for _, name in build._iter_allowlist()]
    assert "banking_v3/environment.py" in names
    assert "banking_v3/generate.py" in names
    assert not any("banking_v3/data/evaluation/" in name or "/evaluator/" in name for name in names)
    assert 'version = "2.0.0"' in build.PYPROJECT
    assert '"banking_v3"' in build.PYPROJECT
