"""Integrity checks for inference artifacts from the qualified repository."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import joblib

TRAFFIC_MODEL = Path("models/traffic_model.joblib")
TRAFFIC_MODEL_SHA256 = "fed0ee3457b8fddc6d386b46652fb9e9e8fc2c058974a5a43ee5ab8e221d1f48"


def verified_bytes(root: Path, relative: Path, expected: str) -> bytes:
    """Resolve a file inside its archive and verify the bytes before use."""
    base = root.resolve()
    path = (base / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(base):
        raise ValueError(f"Artifact path escapes its archive: {relative.name}")
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Artifact integrity check failed: {path.name}")
    return data


def load_traffic_model():
    """Load the qualified phase-five predictor after verifying its digest."""
    return joblib.load(BytesIO(verified_bytes(Path.cwd(), TRAFFIC_MODEL, TRAFFIC_MODEL_SHA256)))
