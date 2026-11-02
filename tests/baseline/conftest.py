import pytest

from tests.baseline.helpers import fake_clip


@pytest.fixture(autouse=True)
def _no_hf_clip(monkeypatch):
    """HiddenObjects: CLIP text giả (test không tải HuggingFace)."""
    monkeypatch.setattr("baseline.methods.hiddenobjects.encode_class_names", fake_clip)
