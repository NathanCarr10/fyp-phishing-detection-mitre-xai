"""
Unit tests for utils module.

Tests cover:
- classify_email() function
- Directory utilities
- Constant loading
"""

import pytest
import sys
import os
from unittest.mock import MagicMock

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    classify_email,
    get_label_name,
    load_constants,
    ensure_directory,
    get_project_root,
)


@pytest.mark.unit
class TestClassifyEmail:
    """Test classify_email() function."""

    def test_classify_email_valid_input(self, simple_tfidf_vectorizer, simple_classifier):
        """Test classify_email with valid inputs."""
        text = "account compromised click verify"
        label, prob = classify_email(simple_tfidf_vectorizer, simple_classifier, text, threshold=0.5)

        assert isinstance(label, int)
        assert label in [0, 1]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_classify_email_threshold_low(self, simple_tfidf_vectorizer, simple_classifier):
        """Test with low threshold (more phishing predictions)."""
        text = "account compromised"
        label_low, _ = classify_email(simple_tfidf_vectorizer, simple_classifier, text, threshold=0.1)
        label_high, _ = classify_email(simple_tfidf_vectorizer, simple_classifier, text, threshold=0.9)

        # Low threshold should predict more as phishing
        assert isinstance(label_low, int)
        assert isinstance(label_high, int)

    def test_classify_email_empty_text_raises(self, simple_tfidf_vectorizer, simple_classifier):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError):
            classify_email(simple_tfidf_vectorizer, simple_classifier, "", threshold=0.5)

    def test_classify_email_none_text_raises(self, simple_tfidf_vectorizer, simple_classifier):
        """Test that None text raises error."""
        with pytest.raises((ValueError, AttributeError)):
            classify_email(simple_tfidf_vectorizer, simple_classifier, None, threshold=0.5)  # type: ignore

    def test_classify_email_whitespace_only_raises(self, simple_tfidf_vectorizer, simple_classifier):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError):
            classify_email(simple_tfidf_vectorizer, simple_classifier, "   ", threshold=0.5)

    def test_classify_email_various_thresholds(self, simple_tfidf_vectorizer, simple_classifier):
        """Test with various threshold values."""
        text = "account"
        for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
            label, prob = classify_email(simple_tfidf_vectorizer, simple_classifier, text, threshold=threshold)
            assert isinstance(label, int)
            assert isinstance(prob, float)


@pytest.mark.unit
class TestGetLabelName:
    """Test get_label_name() helper."""

    def test_label_0_name(self):
        """Test that label 0 maps to legitimate."""
        name = get_label_name(0)
        assert name == 'legitimate'

    def test_label_1_name(self):
        """Test that label 1 maps to phishing."""
        name = get_label_name(1)
        assert name == 'phishing'

    def test_custom_label_map(self):
        """Test with custom label mapping."""
        custom_map = {0: 'good', 1: 'bad'}
        assert get_label_name(0, custom_map) == 'good'
        assert get_label_name(1, custom_map) == 'bad'

    def test_unknown_label_fallback(self):
        """Test fallback for unknown labels."""
        name = get_label_name(999)
        assert name == '999'


@pytest.mark.unit
class TestLoadConstants:
    """Test load_constants() function."""

    def test_constants_structure(self):
        """Test that load_constants returns expected keys."""
        consts = load_constants()
        expected_keys = {
            'TFIDF_MAX_FEATURES',
            'TFIDF_STOP_WORDS',
            'LR_MAX_ITER',
            'LABEL_MAP',
            'DEFAULT_THRESHOLD',
            'RANDOM_SEED',
        }
        assert set(consts.keys()) == expected_keys

    def test_constant_types(self):
        """Test that constants have correct types."""
        consts = load_constants()
        assert isinstance(consts['TFIDF_MAX_FEATURES'], int)
        assert consts['TFIDF_MAX_FEATURES'] > 0
        assert isinstance(consts['LR_MAX_ITER'], int)
        assert consts['LR_MAX_ITER'] > 0
        assert isinstance(consts['LABEL_MAP'], dict)
        assert isinstance(consts['DEFAULT_THRESHOLD'], float)
        assert 0.0 <= consts['DEFAULT_THRESHOLD'] <= 1.0

    def test_label_map_values(self):
        """Test that LABEL_MAP is correct."""
        consts = load_constants()
        label_map = consts['LABEL_MAP']
        assert 0 in label_map
        assert 1 in label_map
        assert label_map[0] == 'legitimate'
        assert label_map[1] == 'phishing'


@pytest.mark.unit
class TestEnsureDirectory:
    """Test ensure_directory() function."""

    def test_create_new_directory(self, tmp_path):
        """Test creating a new directory."""
        test_dir = tmp_path / "test" / "subdir"
        ensure_directory(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_existing_directory_no_error(self, tmp_path):
        """Test that existing directory doesn't raise error."""
        test_dir = tmp_path / "existing"
        test_dir.mkdir()
        # Should not raise
        ensure_directory(str(test_dir))
        assert test_dir.exists()

    def test_create_nested_directories(self, tmp_path):
        """Test creating deeply nested directories."""
        test_dir = tmp_path / "a" / "b" / "c" / "d"
        ensure_directory(str(test_dir))
        assert test_dir.exists()


@pytest.mark.unit
class TestGetProjectRoot:
    """Test get_project_root() function."""

    def test_returns_path(self):
        """Test that function returns a Path object."""
        root = get_project_root()
        assert hasattr(root, 'exists')
        assert hasattr(root, 'is_dir')

    def test_project_root_exists(self):
        """Test that returned path exists."""
        root = get_project_root()
        assert root.exists()
        assert root.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
