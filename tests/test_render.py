from pathlib import Path

import pytest

from datashader_demos import DEMOS, render_demo, validate_png


def test_reference_catalog_has_breadth() -> None:
    assert len(DEMOS) >= 12
    assert len({demo.subtitle for demo in DEMOS.values()}) >= 8


@pytest.mark.parametrize("name", sorted(DEMOS))
def test_demo_renders_transparent_png(name: str, tmp_path: Path) -> None:
    path = render_demo(name, tmp_path, rows=28_000)
    stats = validate_png(path)
    assert path.name.endswith("-transparent.png")
    assert stats.width == 1600
    assert stats.height == 1000
    assert stats.transparent_pixels > 400_000
    assert path.with_suffix(".json").is_file()
