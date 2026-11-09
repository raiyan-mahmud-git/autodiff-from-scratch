import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_node(script: str, *args: str) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for JavaScript demo verification")
    result = subprocess.run(
        [node, script, *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout


def test_javascript_forward_matches_python_export_vectors() -> None:
    output = _run_node("web/verify_forward.js", "web/model.json", "web/test_vectors.json")
    assert "parity passed" in output


def test_digit_segmentation_splits_wide_gaps_and_merges_small_gaps() -> None:
    output = _run_node("web/verify_segmentation.js")
    assert "segmentation checks passed" in output


def test_demo_is_self_contained_and_has_accessible_alternatives() -> None:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert 'id="model-data" type="application/json">{' in html
    assert 'src="app.js?v=3"' in html
    assert 'src="forward.js?v=2"' in html
    assert "<script>" not in html
    assert 'aria-live="polite"' in html
    assert 'data-sample="0"' in html
    assert 'data-sample="1"' in html
    assert 'data-sample="2"' in html
    assert ":focus-visible" in html
    assert "clear space between them" in html
    assert "Hand-drawn digits differ" in html
