"""Test fusion-datasets."""

import fusion_datasets


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(fusion_datasets.__name__, str)
