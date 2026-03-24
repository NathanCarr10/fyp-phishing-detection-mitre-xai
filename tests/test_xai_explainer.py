"""
Unit tests for xai_explainer module.

Tests cover:
- LIME explanations
- Linear weight explanations
- Fallback behavior
- explain_email() public API
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src/ to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xai_explainer import (
    explain_email,
    _explain_with_linear_weights,
    _get_phishing_class_index,
)


@pytest.mark.unit
class TestGetPhishingClassIndex:
    """Test _get_phishing_class_index function."""

    def test_phishing_class_present(self, simple_classifier):
        """Test when phishing class (1) is present in clf.classes_."""
        index = _get_phishing_class_index(simple_classifier)
        assert isinstance(index, int)
        assert index in [0, 1]

    def test_fallback_when_class_missing(self):
        """Test fallback to index 1 when class not found."""
        mock_clf = MagicMock()
        mock_clf.classes_ = [0]  # Only legitimate class
        index = _get_phishing_class_index(mock_clf)
        assert index == 1  # Falls back to 1


@pytest.mark.unit
class TestExplainWithLinearWeights:
    """Test linear weight explanation method."""

    @pytest.mark.requires_model
    def test_returns_list_of_tuples(self, simple_tfidf_vectorizer, simple_classifier):
        """Test that _explain_with_linear_weights returns list of (term, weight) tuples."""
        with patch('xai_explainer._get_model', return_value=(simple_tfidf_vectorizer, simple_classifier)):
            text = "account compromised click verify"
            result = _explain_with_linear_weights(text, num_features=3)

            assert isinstance(result, list)
            assert len(result) <= 3
            for term, weight in result:
                assert isinstance(term, str)
                assert isinstance(weight, float)

    @pytest.mark.requires_model
    def test_num_features_limit(self, simple_tfidf_vectorizer, simple_classifier):
        """Test that result respects num_features limit."""
        with patch('xai_explainer._get_model', return_value=(simple_tfidf_vectorizer, simple_classifier)):
            text = "account compromised click verify identity"
            for n_features in [1, 3, 5, 10]:
                result = _explain_with_linear_weights(text, num_features=n_features)
                assert len(result) <= n_features

    @pytest.mark.requires_model
    def test_empty_email(self, simple_tfidf_vectorizer, simple_classifier):
        """Test handling of emails with no matching features."""
        with patch('xai_explainer._get_model', return_value=(simple_tfidf_vectorizer, simple_classifier)):
            text = "xyz qwerty asdf"  # Unlikely to match training vocab
            result = _explain_with_linear_weights(text, num_features=5)
            # Should return empty or very short list
            assert isinstance(result, list)


@pytest.mark.unit
class TestExplainEmailAPI:
    """Test the main explain_email() public API."""

    @pytest.mark.requires_model
    def test_explain_email_structure(self, sample_emails, simple_tfidf_vectorizer, simple_classifier):
        """Test that explain_email returns correct dict structure."""
        with patch('xai_explainer._get_model', return_value=(simple_tfidf_vectorizer, simple_classifier)):
            text = sample_emails['phishing'][0]
            result = explain_email(text, num_features=5, threshold=0.5, use_lime=False)

            # Check structure
            expected_keys = {'text', 'pred_label', 'phishing_probability', 'threshold',
                           'is_phishing', 'top_features', 'method'}
            assert set(result.keys()) == expected_keys

            # Check types
            assert isinstance(result['text'], str)
            assert isinstance(result['pred_label'], int)
            assert isinstance(result['phishing_probability'], float)
            assert isinstance(result['threshold'], float)
            assert isinstance(result['is_phishing'], bool)
            assert isinstance(result['top_features'], list)
            assert result['method'] in {'lime', 'linear', 'none'}

    @pytest.mark.requires_model
    def test_threshold_affects_prediction(self, simple_tfidf_vectorizer, simple_classifier):
        """Test that different thresholds can change predictions."""
        with patch('xai_explainer._get_model', return_value=(simple_tfidf_vectorizer, simple_classifier)):
            text = "account compromised verify"

            # Get predictions with different thresholds
            result_low = explain_email(text, threshold=0.1)
            result_high = explain_email(text, threshold=0.9)

            # Predictions might differ
            assert isinstance(result_low['is_phishing'], bool)
            assert isinstance(result_high['is_phishing'], bool)

    @pytest.mark.requires_model
    def test_uses_linear_weights_when_lime_unavailable(self, simple_tfidf_vectorizer, simple_classifier):
        """Test fallback to linear weights when LIME unavailable."""
        with patch('xai_explainer._get_model', return_value=(simple_tfidf_vectorizer, simple_classifier)):
            with patch('xai_explainer._LIME_AVAILABLE', False):
                text = "account compromised"
                result = explain_email(text, use_lime=True)
                assert result['method'] == 'linear'

    @pytest.mark.requires_model
    def test_respects_use_lime_false(self, simple_tfidf_vectorizer, simple_classifier):
        """Test that use_lime=False forces linear weights."""
        with patch('xai_explainer._get_model', return_value=(simple_tfidf_vectorizer, simple_classifier)):
            text = "account compromised"
            result = explain_email(text, use_lime=False)
            assert result['method'] == 'linear'

    @pytest.mark.requires_model
    def test_num_features_parameter(self, simple_tfidf_vectorizer, simple_classifier):
        """Test that num_features parameter is respected."""
        with patch('xai_explainer._get_model', return_value=(simple_tfidf_vectorizer, simple_classifier)):
            text = "account compromised click verify identity password"
            for n_features in [1, 3, 5]:
                result = explain_email(text, num_features=n_features, use_lime=False)
                assert len(result['top_features']) <= n_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
