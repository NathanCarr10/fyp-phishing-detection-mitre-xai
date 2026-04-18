"""
Unit tests for Stage 2/3/4 helper functions.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate_models_rigorously import confidence_interval_from_samples
from evaluate_mitre_mapping import extract_technique_id, parse_expected_all, evaluate_multi_label_micro
from run_error_analysis import tokenize, top_tokens, patch_legacy_logreg


@pytest.mark.unit
def test_confidence_interval_from_samples_basic():
    mean, std, ci_low, ci_high = confidence_interval_from_samples([0.2, 0.4, 0.6, 0.8])
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0
    assert ci_low <= ci_high


@pytest.mark.unit
def test_extract_technique_id():
    assert extract_technique_id("T1566.002 - Phishing: Link") == "T1566.002"


@pytest.mark.unit
def test_parse_expected_all():
    parsed = parse_expected_all("T1566.002; T1598.001; ")
    assert parsed == {"T1566.002", "T1598.001"}


@pytest.mark.unit
def test_evaluate_multi_label_micro():
    rows = [
        {
            "expected_all_set": {"T1566.002", "T1598.001"},
            "predicted_all_set": {"T1566.002"},
        },
        {
            "expected_all_set": {"T1566.001"},
            "predicted_all_set": {"T1566.001", "T1598.003"},
        },
    ]
    p, r, f1 = evaluate_multi_label_micro(rows, labels=["T1566.001", "T1566.002", "T1598.001", "T1598.003"])
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0


@pytest.mark.unit
def test_tokenize_and_top_tokens():
    tokens = tokenize("Urgent click here now!")
    assert "urgent" in tokens
    top = top_tokens(["urgent urgent click", "click verify"], top_n=2)
    assert len(top) <= 2
    assert top[0][0] in {"urgent", "click"}


@pytest.mark.unit
class DummyClf:
    pass


@pytest.mark.unit
def test_patch_legacy_logreg_adds_multi_class():
    clf = DummyClf()
    assert not hasattr(clf, "multi_class")
    patch_legacy_logreg(clf)
    assert hasattr(clf, "multi_class")
    assert clf.multi_class == "auto"
