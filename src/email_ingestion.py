"""
Helpers for reading .eml files inside the Streamlit app.

This module safely parses local email files, extracts useful fields,
and passes the text to the existing prediction, MITRE mapping,
and explanation functions used in the project.
"""

from __future__ import annotations

import html
import re
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses
from pathlib import Path
from typing import Any, Dict, List

from attacker_sim import mitre_mapping
from mvp_baseline import load_model
from utils import classify_email
from xai_explainer import explain_email


def _clean_text(value: str) -> str:
    """Normalize whitespace and return a readable string."""
    return " ".join((value or "").split()).strip()


def _html_to_text(html_body: str) -> str:
    """Convert HTML to simple safe readable text without remote/resource loading."""
    text = html_body or ""

    # Remove script/style blocks and HTML comments.
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--[\s\S]*?-->", " ", text)

    # Add line breaks for common block tags before stripping all tags.
    text = re.sub(r"</?(p|div|br|li|tr|h[1-6])\b[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode HTML entities and normalize whitespace.
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def _decode_part_payload(part, payload: bytes | None, charset: str) -> str:
    """Decode one MIME part payload into text with safe fallbacks."""
    if payload is None:
        part_text = part.get_payload()
        return part_text if isinstance(part_text, str) else ""

    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _extract_plain_body(msg) -> str:
    """Extract plain-text body, or fallback to HTML-to-text conversion."""
    plain_parts: List[str] = []
    html_parts: List[str] = []

    for part in msg.walk():
        if part.is_multipart():
            continue

        content_type = (part.get_content_type() or "").lower()
        disposition = (part.get_content_disposition() or "").lower()

        # Skip attachments for body extraction.
        if disposition == "attachment":
            continue

        payload = part.get_payload(decode=True)
        charset = part.get_content_charset() or "utf-8"

        text = _decode_part_payload(part, payload, charset)

        if not text.strip():
            continue

        if content_type == "text/plain":
            plain_parts.append(text)
        elif content_type == "text/html":
            html_parts.append(text)

    if plain_parts:
        return "\n\n".join(p.strip() for p in plain_parts if p.strip())

    if html_parts:
        return "\n\n".join(_html_to_text(p) for p in html_parts if p.strip())

    return ""


def _extract_attachment_names(msg) -> List[str]:
    """Extract attachment filenames only; does not process content."""
    names: List[str] = []

    for part in msg.walk():
        if part.is_multipart():
            continue

        filename = part.get_filename()
        if not filename:
            continue

        # Keep the filename text only, do not read attachment bytes.
        name = Path(str(filename)).name.strip()
        if name:
            names.append(name)

    # Stable order + deduplication.
    seen = set()
    unique_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    return unique_names


def _extract_address_field(msg, header_name: str) -> str:
    """Extract and normalize addresses from a given header."""
    values = msg.get_all(header_name, [])
    parsed = getaddresses(values)

    addresses = []
    for display_name, address in parsed:
        display_name = _clean_text(display_name)
        address = _clean_text(address)
        if display_name and address:
            addresses.append(f"{display_name} <{address}>")
        elif address:
            addresses.append(address)
        elif display_name:
            addresses.append(display_name)

    if addresses:
        return ", ".join(addresses)

    # Fallback to raw header if parsing did not return usable tokens.
    return _clean_text(str(msg.get(header_name, "")))


def parse_eml_file(file_bytes: bytes) -> Dict[str, Any]:
    """Parse a .eml file and pull out the fields I need."""
    if not isinstance(file_bytes, (bytes, bytearray)) or not file_bytes:
        raise ValueError("Uploaded .eml content is empty or invalid.")

    msg = BytesParser(policy=policy.default).parsebytes(bytes(file_bytes))

    sender = _extract_address_field(msg, "from")
    recipients = _extract_address_field(msg, "to")
    subject = _clean_text(str(msg.get("subject", "")))
    date = _clean_text(str(msg.get("date", "")))
    body = _extract_plain_body(msg)
    attachment_names = _extract_attachment_names(msg)

    combined_parts = [
        f"From: {sender}" if sender else "",
        f"To: {recipients}" if recipients else "",
        f"Subject: {subject}" if subject else "",
        body,
    ]
    combined_text = "\n".join(part for part in combined_parts if part).strip()

    return {
        "from": sender,
        "to": recipients,
        "subject": subject,
        "date": date,
        "body": body,
        "attachment_names": attachment_names,
        "combined_text": combined_text,
    }


def analyze_combined_text(
    combined_text: str,
    threshold: float = 0.5,
    num_features: int = 10,
    use_lime: bool = True,
    use_shap: bool = False,
    vectorizer=None,
    clf=None,
) -> Dict[str, Any]:
    """Run the model, MITRE mapping, and explanation on one email."""
    if not isinstance(combined_text, str) or not combined_text.strip():
        raise ValueError("combined_text must be a non-empty string.")

    if vectorizer is None or clf is None:
        vectorizer, clf = load_model()
    pred_label, phishing_probability = classify_email(
        vectorizer,
        clf,
        combined_text,
        threshold=threshold,
    )

    try:
        xai_explanation = explain_email(
            combined_text,
            num_features=num_features,
            threshold=threshold,
            use_lime=use_lime,
            use_shap=use_shap,
        )
    except Exception as exc:
        xai_explanation = {
            "method": f"⚠️ Explanation Error: {str(exc)}",
            "top_features": [],
        }

    return {
        "predicted_label": pred_label,
        "phishing_probability": phishing_probability,
        "mitre_mapping": mitre_mapping(combined_text),
        "xai_explanation": xai_explanation,
    }
