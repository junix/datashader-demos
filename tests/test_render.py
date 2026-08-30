import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from datashader_demos import DEMOS, render_demo, validate_png
from datashader_demos.render import PixelStats


def _write_png(path: Path, rgba: np.ndarray) -> Path:
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")
    return path


def test_reference_catalog_has_breadth() -> None:
    assert len(DEMOS) == 12
    assert len({demo.subtitle for demo in DEMOS.values()}) == len(DEMOS)


def test_reference_catalog_entries_are_self_consistent() -> None:
    assert list(DEMOS) == [demo.name for demo in DEMOS.values()]
    assert len({demo.title for demo in DEMOS.values()}) == len(DEMOS)
    for demo in DEMOS.values():
        assert demo.title == demo.title.strip()
        assert demo.subtitle == demo.subtitle.strip()
        assert demo.default_rows > 0
        assert callable(demo.renderer)


@pytest.mark.parametrize("name", sorted(DEMOS))
def test_demo_renders_transparent_png(name: str, tmp_path: Path) -> None:
    path = render_demo(name, tmp_path, rows=28_000)
    stats = validate_png(path)
    assert path == tmp_path / f"{name}-transparent.png"
    with Image.open(path) as rendered:
        assert rendered.format == "PNG"
        assert rendered.mode == "RGBA"
    assert stats.width == 1600
    assert stats.height == 1000
    assert stats.transparent_pixels > 400_000
    raw_manifest = path.with_suffix(".json").read_text(encoding="utf-8")
    manifest = json.loads(raw_manifest)
    assert set(manifest) == {
        "demo",
        "title",
        "requested_rows",
        "render_seconds",
        "artifact",
        "pixel_stats",
        "background",
        "data",
    }
    demo = DEMOS[name]
    assert manifest["demo"] == demo.name
    assert manifest["title"] == demo.title
    assert manifest["requested_rows"] == 28_000
    assert manifest["artifact"] == path.name
    assert manifest["background"] == "transparent"
    assert manifest["data"] == "deterministic synthetic fixture"
    assert manifest["pixel_stats"] == asdict(stats)
    # manifest is written as json.dumps(..., indent=2) plus a trailing newline
    assert raw_manifest == json.dumps(manifest, indent=2) + "\n"
    assert manifest["render_seconds"] >= 0.0


def test_render_demo_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="'no-such-demo'"):
        render_demo("no-such-demo", tmp_path, rows=1_000)
    assert list(tmp_path.iterdir()) == []


def test_render_demo_uses_catalog_default_rows(tmp_path: Path) -> None:
    path = render_demo("phase-portrait", tmp_path, rows=None)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["requested_rows"] == DEMOS["phase-portrait"].default_rows


def test_render_demo_treats_zero_rows_as_catalog_default(tmp_path: Path) -> None:
    # rows=0 is falsy, so `rows or demo.default_rows` silently falls back to the default.
    path = render_demo("phase-portrait", tmp_path, rows=0)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["requested_rows"] == DEMOS["phase-portrait"].default_rows


def test_render_demo_is_deterministic(tmp_path: Path) -> None:
    first = render_demo("point-density", tmp_path / "nested" / "deep", rows=28_000)
    second = render_demo("point-density", tmp_path / "flat", rows=28_000)
    assert first.parent == tmp_path / "nested" / "deep"
    assert first.read_bytes() == second.read_bytes()


def test_validate_png_counts_alpha_bands_and_color_channels(tmp_path: Path) -> None:
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    rgba[:, :50] = (0, 0, 0, 0)  # alpha 0: transparent (10_000 px = exactly the 25% floor)
    rgba[:, 50:80] = (10, 20, 30, 10)  # alpha 8..24 dead band: neither transparent nor visible
    rgba[:, 80:140] = (128, 128, 128, 200)  # visible but monochrome: not colorful
    rgba[:, 140:] = (255, 0, 0, 255)  # visible and colorful
    path = _write_png(tmp_path / "bands.png", rgba)
    assert validate_png(path) == PixelStats(
        width=200,
        height=200,
        transparent_pixels=10_000,
        visible_pixels=24_000,
        colorful_pixels=12_000,
    )


def test_validate_png_pins_every_threshold_exactly(tmp_path: Path) -> None:
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    rgba[:, 0:50] = (200, 10, 10, 7)  # alpha 7 < 8: transparent even with vivid rgb
    rgba[:, 50:75] = (200, 10, 10, 8)  # alpha 8: first dead-band value, not transparent
    rgba[:, 75:100] = (200, 10, 10, 16)  # mid dead band: neither transparent nor visible
    rgba[:, 100:125] = (200, 10, 10, 24)  # alpha 24: last dead-band value, not visible
    rgba[:, 125:150] = (100, 100, 100, 25)  # alpha 25: first visible; chroma 0, not colorful
    rgba[:, 150:162] = (200, 10, 10, 64)  # alpha 64: visible but not > 64, so not colorful
    rgba[:, 162:175] = (0, 0, 28, 65)  # chroma 28: not > 28, so not colorful
    rgba[:, 175:] = (0, 0, 29, 65)  # chroma 29 with alpha 65: colorful
    path = _write_png(tmp_path / "thresholds.png", rgba)
    # 10_000/15_000/5_000 sit exactly on the three coverage floors (== passes, < raises).
    assert validate_png(path) == PixelStats(
        width=200,
        height=200,
        transparent_pixels=10_000,
        visible_pixels=15_000,
        colorful_pixels=5_000,
    )


def test_validate_png_rejects_one_column_below_transparent_floor(tmp_path: Path) -> None:
    rgba = np.full((200, 200, 4), (255, 0, 0, 255), dtype=np.uint8)
    rgba[:, :49] = (255, 0, 0, 7)  # 49 columns: 9_800 transparent < 10_000 floor (25% of 40_000)
    path = _write_png(tmp_path / "short.png", rgba)
    with pytest.raises(ValueError, match="transparent background coverage is too small") as excinfo:
        validate_png(path)
    assert str(path) in str(excinfo.value)


def test_validate_png_rejects_opaque_background(tmp_path: Path) -> None:
    rgba = np.full((50, 100, 4), (255, 0, 0, 255), dtype=np.uint8)
    path = _write_png(tmp_path / "opaque.png", rgba)
    with pytest.raises(ValueError, match="transparent background coverage is too small") as excinfo:
        validate_png(path)
    assert str(path) in str(excinfo.value)


def test_validate_png_rejects_blank_canvas(tmp_path: Path) -> None:
    path = _write_png(tmp_path / "blank.png", np.zeros((50, 100, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="visible plot content is too small") as excinfo:
        validate_png(path)
    assert str(path) in str(excinfo.value)


def test_validate_png_rejects_colorless_content(tmp_path: Path) -> None:
    rgba = np.zeros((150, 200, 4), dtype=np.uint8)
    rgba[:, 80:] = (128, 128, 128, 255)
    path = _write_png(tmp_path / "gray.png", rgba)
    with pytest.raises(ValueError, match="colorful data pixels are missing") as excinfo:
        validate_png(path)
    assert str(path) in str(excinfo.value)
