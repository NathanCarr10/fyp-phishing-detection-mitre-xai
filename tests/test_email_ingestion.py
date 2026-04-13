"""
Unit tests for email_ingestion module.

Tests cover:
- .eml parsing
- HTML-to-text fallback
- attachment filename extraction
- analysis helper integration with existing project functions
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from email_ingestion import parse_eml_file, analyze_combined_text


@pytest.mark.unit
class TestParseEmlFile:
    """Test parse_eml_file() function."""

    def test_parse_simple_plain_text_eml(self):
        """Test parsing a simple plain-text .eml message."""
        eml_bytes = b"""From: Alice <alice@example.com>
To: Bob <bob@example.com>
Subject: Test Email
Date: Mon, 1 Apr 2026 10:00:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Hello Bob,
Please review the attached report.
Thanks,
Alice
"""

        result = parse_eml_file(eml_bytes)

        assert result["from"] == "Alice <alice@example.com>"
        assert result["to"] == "Bob <bob@example.com>"
        assert result["subject"] == "Test Email"
        assert "Apr 2026" in result["date"]
        assert "10:00:00 +0000" in result["date"]
        assert "Hello Bob" in result["body"]
        assert result["attachment_names"] == []
        assert "From: Alice <alice@example.com>" in result["combined_text"]
        assert "Subject: Test Email" in result["combined_text"]
        assert "Please review the attached report." in result["combined_text"]

    def test_parse_html_only_eml(self):
        """Test HTML-only emails are converted to readable text."""
        eml_bytes = b"""From: Sender <sender@example.com>
To: Recipient <recipient@example.com>
Subject: HTML Only
MIME-Version: 1.0
Content-Type: text/html; charset=utf-8

<html>
  <head>
    <style>body { color: red; }</style>
    <script>alert('x');</script>
  </head>
  <body>
    <p>Dear user,</p>
    <div>Please <b>verify</b> your account.</div>
    <p>Click here to continue.</p>
  </body>
</html>
"""

        result = parse_eml_file(eml_bytes)

        assert result["body"]
        assert "verify your account" in result["body"].lower()
        assert "click here to continue" in result["body"].lower()
        assert "<script>" not in result["body"].lower()
        assert "<style>" not in result["body"].lower()
        assert "<" not in result["body"]

    def test_parse_attachment_filenames_only(self):
        """Test that attachments are reported by filename only."""
        eml_bytes = b"""From: Sender <sender@example.com>
To: Recipient <recipient@example.com>
Subject: With Attachment
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="BOUNDARY"

--BOUNDARY
Content-Type: text/plain; charset=utf-8

Please see attached.
--BOUNDARY
Content-Type: application/pdf
Content-Disposition: attachment; filename="invoice.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjQKJcfs...
--BOUNDARY--
"""

        result = parse_eml_file(eml_bytes)

        assert result["attachment_names"] == ["invoice.pdf"]
        assert "invoice.pdf" not in result["combined_text"]

    def test_parse_empty_bytes_raises(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError):
            parse_eml_file(b"")


@pytest.mark.unit
class TestAnalyzeCombinedText:
    """Test analyze_combined_text() function."""

    def test_analysis_helper_uses_existing_project_functions(self):
        """Test helper returns expected structure and calls existing functions."""
        fake_vectorizer = MagicMock()
        fake_clf = MagicMock()

        with patch("email_ingestion.load_model", return_value=(fake_vectorizer, fake_clf)) as load_model_mock, \
             patch("email_ingestion.classify_email", return_value=(1, 0.91)) as classify_mock, \
             patch("email_ingestion.mitre_mapping", return_value="T1566.002 - Phishing: Link") as mitre_mock, \
             patch("email_ingestion.explain_email", return_value={
                 "method": "linear",
                 "top_features": [{"term": "verify", "weight": 0.12}],
             }) as explain_mock:

            result = analyze_combined_text(
                "From: Alice <alice@example.com>\nSubject: Please verify",
                threshold=0.6,
                num_features=5,
                use_lime=False,
                vectorizer=fake_vectorizer,
                clf=fake_clf,
            )

        load_model_mock.assert_not_called()
        classify_mock.assert_called_once()
        mitre_mock.assert_called_once()
        explain_mock.assert_called_once()

        assert result["predicted_label"] == 1
        assert result["phishing_probability"] == 0.91
        assert result["mitre_mapping"] == "T1566.002 - Phishing: Link"
        assert result["xai_explanation"]["method"] == "linear"
        assert result["xai_explanation"]["top_features"][0]["term"] == "verify"

    def test_analysis_helper_loads_model_when_not_provided(self):
        """Test helper loads the model if vectorizer/clf are omitted."""
        fake_vectorizer = MagicMock()
        fake_clf = MagicMock()

        with patch("email_ingestion.load_model", return_value=(fake_vectorizer, fake_clf)) as load_model_mock, \
             patch("email_ingestion.classify_email", return_value=(0, 0.12)), \
             patch("email_ingestion.mitre_mapping", return_value="T1566.001 - Phishing: Attachment"), \
             patch("email_ingestion.explain_email", return_value={"method": "linear", "top_features": []}):

            result = analyze_combined_text("Subject: Hello\nBody text", use_lime=False)

        load_model_mock.assert_called_once()
        assert result["predicted_label"] == 0
        assert result["phishing_probability"] == 0.12

    def test_analysis_helper_empty_text_raises(self):
        """Test that empty combined text raises ValueError."""
        with pytest.raises(ValueError):
            analyze_combined_text("   ")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
