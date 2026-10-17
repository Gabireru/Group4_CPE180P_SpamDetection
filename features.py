"""Shared input-feature statistics used for data-drift monitoring."""
import re

FEATURES = ["length", "word_count", "digit_ratio", "upper_ratio"]


def msg_features(text: str) -> dict:
    n = max(len(text), 1)
    return {
        "length": float(len(text)),
        "word_count": float(len(text.split())),
        "digit_ratio": sum(c.isdigit() for c in text) / n,
        "upper_ratio": sum(c.isupper() for c in text) / n,
    }


def normalize(text: str) -> str:
    """Shared cleaning used in training AND serving (avoids train/serve skew).

    Lowercasing is done here instead of inside the ONNX graph because the ONNX
    StringNormalizer op needs an en_US.UTF-8 locale, which python:3.10-slim lacks.
    """
    return re.sub(r"\s+", " ", text).strip().lower()
