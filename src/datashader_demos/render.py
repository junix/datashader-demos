"""Deterministic Datashader examples with presentation-ready transparent output."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

WIDTH = 1600
HEIGHT = 1000
PLOT_WIDTH = 1420
PLOT_HEIGHT = 760
PLOT_OFFSET = (90, 170)


@dataclass(frozen=True)
class Demo:
    name: str
    title: str
    subtitle: str
    default_rows: int
    renderer: Callable[[int], Image.Image]


@dataclass(frozen=True)
class PixelStats:
    width: int
    height: int
    transparent_pixels: int
    visible_pixels: int
    colorful_pixels: int


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _decorate(
    raster: Image.Image,
    *,
    title: str,
    subtitle: str,
    rows: int,
    legend: list[tuple[str, str]],
) -> Image.Image:
    output = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    output.alpha_composite(raster.convert("RGBA"), PLOT_OFFSET)
    draw = ImageDraw.Draw(output)
    ink = (26, 42, 66, 255)
    muted = (76, 93, 112, 235)
    accent = (181, 67, 98, 255)
    draw.text((90, 48), "DATASHADER · RASTER AGGREGATION", font=_font(17, bold=True), fill=accent)
    draw.text((90, 76), title, font=_font(44, bold=True), fill=ink)
    draw.text((92, 131), subtitle, font=_font(18), fill=muted)
    draw.text((1510, 58), f"{rows:,} ROWS", font=_font(16, bold=True), fill=ink, anchor="ra")
    legend_x = 94
    legend_font = _font(15)
    for label, color in legend:
        draw.rounded_rectangle((legend_x, 958, legend_x + 24, 974), radius=5, fill=color)
        draw.text((legend_x + 33, 953), label, font=legend_font, fill=muted)
        legend_x += 48 + int(draw.textlength(label, font=legend_font))
    draw.text(
        (1510, 953),
        "synthetic deterministic fixture",
        font=_font(14),
        fill=muted,
        anchor="ra",
    )
    return output


def _point_density(rows: int) -> Image.Image:
    rng = np.random.default_rng(42)
    groups = rng.choice(5, size=rows, p=[0.24, 0.20, 0.19, 0.22, 0.15])
    centers = np.asarray([[-3.8, 1.0], [-1.3, -1.7], [1.2, 1.8], [3.8, -0.4], [0.3, 0.1]])
    scales = np.asarray([[0.9, 1.8], [1.4, 0.65], [1.0, 1.1], [0.75, 1.55], [2.1, 0.45]])
    xy = centers[groups] + rng.normal(size=(rows, 2)) * scales[groups]
    xy[:, 1] += 0.45 * np.sin(xy[:, 0] * 1.7)
    frame = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH,
        plot_height=PLOT_HEIGHT,
        x_range=(-7.5, 7.5),
        y_range=(-5.2, 5.2),
    )
    agg = canvas.points(frame, "x", "y", agg=ds.count())
    raster = tf.shade(
        agg,
        cmap=["#162a5f", "#2456a6", "#168fa2", "#57c785", "#f2d35d", "#ef6a55"],
        how="eq_hist",
        min_alpha=25,
    )
    raster = tf.dynspread(raster, threshold=0.72, max_px=2).to_pil()
    return _decorate(
        raster,
        title="Constellation of concentrated demand",
        subtitle="Point density reveals five overlapping regimes without drawing individual marks.",
        rows=rows,
        legend=[
            ("sparse", "#2456a6"),
            ("active", "#168fa2"),
            ("dense", "#f2d35d"),
            ("peak", "#ef6a55"),
        ],
    )


def _trajectory_bundle(rows: int) -> Image.Image:
    steps = 82
    paths = max(20, rows // steps)
    actual_rows = paths * (steps + 1)
    path_id = np.repeat(np.arange(paths), steps)
    t = np.tile(np.linspace(0, 1, steps), paths)
    phase = (path_id * 0.61803398875) % 1 * np.pi * 2
    lane = ((path_id % 17) - 8) * 0.09
    direction = np.where(path_id % 2 == 0, 1.0, -1.0)
    span = 7.2 + (path_id % 23) * 0.27
    center = (((path_id * 37) % 101) / 100.0 - 0.5) * 2.4
    x = center + direction * (t - 0.5) * span + np.sin(t * np.pi * 3 + phase) * 0.75
    y = np.sin(t * np.pi * 2 + phase) * (1.8 + (path_id % 5) * 0.16) + lane
    x += 0.7 * np.sin(y * 1.3)
    # NaN separators prevent Datashader from connecting the end of one path to the next.
    x = np.column_stack((x.reshape(paths, steps), np.full(paths, np.nan))).ravel()
    y = np.column_stack((y.reshape(paths, steps), np.full(paths, np.nan))).ravel()
    frame = pd.DataFrame({"x": x, "y": y})
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH,
        plot_height=PLOT_HEIGHT,
        x_range=(-8, 8),
        y_range=(-4.3, 4.3),
    )
    agg = canvas.line(frame, "x", "y", agg=ds.count(), axis=0)
    raster = tf.shade(
        agg,
        cmap=["#16304f", "#1d6a8a", "#20a39e", "#90d743", "#f4c95d", "#f25f5c"],
        how="eq_hist",
        min_alpha=18,
    )
    raster = tf.dynspread(raster, threshold=0.68, max_px=2).to_pil()
    return _decorate(
        raster,
        title="Braided trajectories through a shared field",
        subtitle=(
            "Thousands of paths accumulate into luminous corridors "
            "while rare detours remain visible."
        ),
        rows=actual_rows,
        legend=[
            ("single path", "#1d6a8a"),
            ("corridor", "#20a39e"),
            ("convergence", "#f4c95d"),
            ("hotspot", "#f25f5c"),
        ],
    )


def _category_mix(rows: int) -> Image.Image:
    rng = np.random.default_rng(84)
    labels = np.asarray(["research", "product", "operations", "community"])
    category_index = rng.integers(0, 4, size=rows)
    angle = rng.uniform(0, np.pi * 2, rows)
    radius = rng.gamma(2.1, 1.0, rows)
    offsets = np.asarray([[-1.9, 1.0], [1.7, 1.2], [-0.9, -1.7], [2.1, -1.4]])
    x = np.cos(angle) * radius + offsets[category_index, 0] + 0.28 * np.sin(radius * 3)
    y = np.sin(angle) * radius * 0.72 + offsets[category_index, 1]
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "category": pd.Categorical(labels[category_index], categories=labels),
        }
    )
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH,
        plot_height=PLOT_HEIGHT,
        x_range=(-7.2, 7.2),
        y_range=(-5.2, 5.2),
    )
    agg = canvas.points(frame, "x", "y", agg=ds.count_cat("category"))
    colors = {
        "research": "#4c78a8",
        "product": "#e45756",
        "operations": "#72b7b2",
        "community": "#f2cf5b",
    }
    raster = tf.shade(agg, color_key=colors, how="eq_hist", min_alpha=24).to_pil()
    return _decorate(
        raster,
        title="Categorical territories and their borderlands",
        subtitle="Color mixing exposes where four populations dominate, overlap, and transition.",
        rows=rows,
        legend=list(colors.items()),
    )


DEMOS: dict[str, Demo] = {
    "point-density": Demo(
        "point-density",
        "Constellation of concentrated demand",
        "Large point aggregation",
        2_000_000,
        _point_density,
    ),
    "trajectory-bundle": Demo(
        "trajectory-bundle",
        "Braided trajectories through a shared field",
        "Path density aggregation",
        500_000,
        _trajectory_bundle,
    ),
    "category-mix": Demo(
        "category-mix",
        "Categorical territories and their borderlands",
        "Categorical point aggregation",
        1_200_000,
        _category_mix,
    ),
}


def validate_png(path: Path) -> PixelStats:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[..., 3]
    rgb = rgba[..., :3].astype(np.int16)
    stats = PixelStats(
        width=rgba.shape[1],
        height=rgba.shape[0],
        transparent_pixels=int(np.count_nonzero(alpha < 8)),
        visible_pixels=int(np.count_nonzero(alpha > 24)),
        colorful_pixels=int(
            np.count_nonzero((alpha > 64) & ((rgb.max(axis=2) - rgb.min(axis=2)) > 28))
        ),
    )
    pixels = stats.width * stats.height
    if stats.transparent_pixels < pixels * 0.25:
        raise ValueError(f"{path}: transparent background coverage is too small")
    if stats.visible_pixels < 15_000:
        raise ValueError(f"{path}: visible plot content is too small")
    if stats.colorful_pixels < 5_000:
        raise ValueError(f"{path}: colorful data pixels are missing")
    return stats


def render_demo(name: str, output_dir: Path, *, rows: int | None = None) -> Path:
    demo = DEMOS[name]
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = rows or demo.default_rows
    start = time.perf_counter()
    image = demo.renderer(row_count)
    path = output_dir / f"{demo.name}-transparent.png"
    image.save(path, format="PNG", optimize=True)
    elapsed = time.perf_counter() - start
    stats = validate_png(path)
    manifest = {
        "demo": demo.name,
        "title": demo.title,
        "requested_rows": row_count,
        "render_seconds": round(elapsed, 4),
        "artifact": path.name,
        "pixel_stats": asdict(stats),
        "background": "transparent",
        "data": "deterministic synthetic fixture",
    }
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
