"""
Shared test fixtures and configuration for pytest.

This module defines reusable fixtures and test data for all test modules.
"""

import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path


@pytest.fixture(scope="session")
def sample_emails():
    """
    Provide sample emails for testing.

    Returns:
        dict: Dictionary with 'legitimate' and 'phishing' email samples
    """
    return {
        'legitimate': [
            "Hi team, please review the attached document for the meeting tomorrow.",
            "The quarterly report is ready for your signature.",
            "Let's schedule a call for next week to discuss the project roadmap.",
            "Thanks for your email. I'll get back to you by end of day.",
            "Here are the updated guidelines for the new policy.",
        ],
        'phishing': [
            "URGENT: Your bank account has been compromised. Click here immediately to verify your identity.",
            "Congratulations! You have won a prize. Click this link to claim: http://phishing-site.com",
            "Your PayPal account will be closed. Please confirm your password: http://fake-paypal.com",
            "ALERT: Unusual login detected. Verify your account now: http://evil.com/verify",
            "IRS Refund Ready: $850 awaiting you. Click to claim: http://fake-irs.com/refund",
        ]
    }


@pytest.fixture(scope="session")
def simple_tfidf_vectorizer():
    """
    Create a simple fitted TF-IDF vectorizer for testing.

    Returns:
        TfidfVectorizer: Fitted vectorizer on sample emails
    """
    emails = [
        "your account has been compromised click here",
        "thanks for your help with the project",
        "urgent verify identity immediately",
        "scheduled meeting tomorrow afternoon",
        "congratulations you have won",
    ]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=100,
    )
    vectorizer.fit(emails)
    return vectorizer


@pytest.fixture(scope="session")
def simple_classifier():
    """
    Create a simple fitted Logistic Regression classifier for testing.

    Returns:
        LogisticRegression: Fitted classifier on sample data
    """
    emails = [
        "your account compromised click verify",
        "thank you for help",
        "urgent action required immediately",
        "meeting scheduled tomorrow",
        "congratulations you won",
    ]
    labels = [1, 0, 1, 0, 1]  # phishing, legit, phishing, legit, phishing

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=100,
    )
    X = vectorizer.fit_transform(emails)

    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X, labels)
    return clf


@pytest.fixture
def project_root():
    """
    Get the project root directory.

    Returns:
        Path: Path to project root (parent of tests/)
    """
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root):
    """
    Get the test data directory.

    Returns:
        Path: Path to data/ directory
    """
    return project_root / "data"


@pytest.fixture
def sample_classification_result():
    """
    Provide a sample classification result dict.

    Returns:
        dict: Result structure from classify_email()
    """
    return {
        'label': 1,
        'label_name': 'phishing',
        'probability': 0.87,
        'probs_dict': {
            'legitimate': 0.13,
            'phishing': 0.87,
        }
    }


@pytest.fixture
def sample_explanation():
    """
    Provide a sample explanation result dict.

    Returns:
        dict: Result structure from explain_email()
    """
    return {
        'text': 'Your account has been compromised. Click here.',
        'pred_label': 1,
        'phishing_probability': 0.92,
        'threshold': 0.5,
        'is_phishing': True,
        'top_features': [
            {'term': 'account', 'weight': 0.15},
            {'term': 'compromised', 'weight': 0.13},
            {'term': 'click', 'weight': 0.11},
        ],
        'method': 'lime',
    }


# Test markers for organizing tests
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring trained model files"
    )
