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


def _phase_portrait(rows: int) -> Image.Image:
    paths = max(80, rows // 160)
    steps = 160
    pid = np.repeat(np.arange(paths), steps)
    t = np.tile(np.linspace(0, 18, steps), paths)
    phase = (pid * 0.618) % 1 * np.pi * 2
    decay = np.exp(-t * (0.018 + (pid % 9) * 0.002))
    x = decay * (3.8 * np.cos(t + phase) + 0.5 * np.cos(3 * t))
    y = decay * (2.7 * np.sin(t * 1.03 + phase) + 0.4 * np.sin(5 * t))
    x = np.column_stack((x.reshape(paths, steps), np.full(paths, np.nan))).ravel()
    y = np.column_stack((y.reshape(paths, steps), np.full(paths, np.nan))).ravel()
    frame = pd.DataFrame({"x": x, "y": y})
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=(-5, 5), y_range=(-4, 4)
    )
    agg = canvas.line(frame, "x", "y", agg=ds.count(), axis=0)
    raster = tf.shade(
        agg,
        cmap=["#152d58", "#315f9d", "#54d6c6", "#ffd166", "#ff6f91"],
        how="eq_hist",
        min_alpha=18,
    ).to_pil()
    return _decorate(
        raster,
        title="Phase-space convergence atlas",
        subtitle=(
            "Thousands of damped trajectories reveal attractors, overshoot, and orbital density."
        ),
        rows=len(frame),
        legend=[("rare", "#315f9d"), ("stable", "#54d6c6"), ("attractor", "#ff6f91")],
    )


def _fractal_basin(rows: int) -> Image.Image:
    side = max(180, int(np.sqrt(rows)))
    xs = np.linspace(-2, 2, side)
    ys = np.linspace(-1.7, 1.7, side)
    x, y = np.meshgrid(xs, ys)
    z = x + 1j * y
    for _ in range(28):
        z -= (z**3 - 1) / (3 * z**2 + 1e-9)
    roots = np.asarray([1 + 0j, -0.5 + 0.866j, -0.5 - 0.866j])
    root = np.argmin(np.abs(z[..., None] - roots), axis=2)
    frame = pd.DataFrame(
        {
            "x": x.ravel(),
            "y": y.ravel(),
            "root": pd.Categorical(np.asarray(["root α", "root β", "root γ"])[root.ravel()]),
        }
    )
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=(-2, 2), y_range=(-1.7, 1.7)
    )
    agg = canvas.points(frame, "x", "y", agg=ds.count_cat("root"))
    colors = {"root α": "#54d6c6", "root β": "#7b9cff", "root γ": "#ff6f91"}
    raster = tf.shade(agg, color_key=colors, how="linear", min_alpha=210).to_pil()
    return _decorate(
        raster,
        title="Newton fractal basins",
        subtitle="Every pixel records which complex root captures the iterative solver.",
        rows=len(frame),
        legend=list(colors.items()),
    )


def _parameter_sweep(rows: int) -> Image.Image:
    side = max(250, int(np.sqrt(rows)))
    a = np.linspace(2.5, 4, side)
    x0 = np.linspace(0.02, 0.98, side)
    aa, xx = np.meshgrid(a, x0)
    state = xx.copy()
    for _ in range(160):
        state = aa * state * (1 - state)
    frame = pd.DataFrame({"a": aa.ravel(), "state": state.ravel()})
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=(2.5, 4), y_range=(0, 1)
    )
    agg = canvas.points(frame, "a", "state", agg=ds.count())
    raster = tf.shade(
        agg,
        cmap=["#152b50", "#315e9e", "#54d6c6", "#ffd166", "#ff6f91"],
        how="eq_hist",
        min_alpha=20,
    )
    raster = tf.dynspread(raster, threshold=0.72, max_px=2).to_pil()
    return _decorate(
        raster,
        title="Logistic-map parameter sweep",
        subtitle="A dense bifurcation diagram exposes stable bands, period doubling, and chaos.",
        rows=len(frame),
        legend=[("stable", "#315e9e"), ("branching", "#54d6c6"), ("chaos", "#ff6f91")],
    )


def _event_raster(rows: int) -> Image.Image:
    rng = np.random.default_rng(420)
    service = rng.integers(0, 42, rows)
    timev = rng.uniform(0, 1440, rows)
    timev += 80 * np.sin(service * 0.7) + rng.normal(0, 18, rows)
    severity = pd.Categorical(
        np.asarray(["info", "warn", "critical"])[
            (rng.random(rows) > 0.82).astype(int) + (rng.random(rows) > 0.97).astype(int)
        ],
        categories=["info", "warn", "critical"],
    )
    frame = pd.DataFrame({"time": timev, "service": service, "severity": severity})
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=(-120, 1560), y_range=(-2, 44)
    )
    agg = canvas.points(frame, "time", "service", agg=ds.count_cat("severity"))
    colors = {"info": "#315e9e", "warn": "#ffd166", "critical": "#ff6f91"}
    raster = tf.shade(agg, color_key=colors, how="eq_hist", min_alpha=22).to_pil()
    return _decorate(
        raster,
        title="Distributed event raster",
        subtitle="Millions of service events expose periodic load, bursts, and critical incidents.",
        rows=rows,
        legend=list(colors.items()),
    )


def _uncertainty_fan(rows: int) -> Image.Image:
    steps = 180
    paths = max(40, rows // steps)
    pid = np.repeat(np.arange(paths), steps)
    t = np.tile(np.linspace(0, 1, steps), paths)
    trend = 1.4 * np.sin(t * 6.2) + 2.2 * t
    spread = (0.1 + 1.2 * t) * (np.sin(pid * 0.73 + t * 13) + 0.5 * np.cos(pid * 0.19 + t * 7))
    y = trend + spread
    t = np.column_stack((t.reshape(paths, steps), np.full(paths, np.nan))).ravel()
    y = np.column_stack((y.reshape(paths, steps), np.full(paths, np.nan))).ravel()
    frame = pd.DataFrame({"t": t, "y": y})
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=(0, 1), y_range=(-3, 5)
    )
    agg = canvas.line(frame, "t", "y", agg=ds.count(), axis=0)
    raster = tf.shade(
        agg,
        cmap=["#172c55", "#35669c", "#54d6c6", "#ffd166", "#ff6f91"],
        how="eq_hist",
        min_alpha=18,
    ).to_pil()
    return _decorate(
        raster,
        title="Monte Carlo uncertainty fan",
        subtitle="Ensemble path density shows the forecast median, spread, and divergent tails.",
        rows=len(frame),
        legend=[("tail", "#35669c"), ("credible band", "#54d6c6"), ("mode", "#ffd166")],
    )


def _network_density(rows: int) -> Image.Image:
    edges = max(200, rows // 45)
    steps = 44
    eid = np.repeat(np.arange(edges), steps)
    t = np.tile(np.linspace(0, 1, steps), edges)
    s = (eid * 37) % 97 / 97 * 2 * np.pi
    target = (eid * 73) % 101 / 101 * 2 * np.pi
    x0 = np.cos(s) * 4.8
    y0 = np.sin(s) * 3.1
    x1 = np.cos(target) * 4.8
    y1 = np.sin(target) * 3.1
    bend = np.sin(t * np.pi) * (1 + (eid % 7) * 0.12)
    x = x0 * (1 - t) + x1 * t - bend * (y1 - y0) * 0.16
    y = y0 * (1 - t) + y1 * t + bend * (x1 - x0) * 0.16
    x = np.column_stack((x.reshape(edges, steps), np.full(edges, np.nan))).ravel()
    y = np.column_stack((y.reshape(edges, steps), np.full(edges, np.nan))).ravel()
    frame = pd.DataFrame({"x": x, "y": y})
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=(-6, 6), y_range=(-4.5, 4.5)
    )
    agg = canvas.line(frame, "x", "y", agg=ds.count(), axis=0)
    raster = tf.shade(
        agg,
        cmap=["#172c55", "#315f9d", "#54d6c6", "#ffd166", "#ff6f91"],
        how="eq_hist",
        min_alpha=12,
    ).to_pil()
    return _decorate(
        raster,
        title="Edge-density network field",
        subtitle=(
            "Tens of thousands of curved connections become corridors instead of hairball clutter."
        ),
        rows=len(frame),
        legend=[("edge", "#315f9d"), ("corridor", "#54d6c6"), ("hub", "#ff6f91")],
    )


def _terrain_scan(rows: int) -> Image.Image:
    rng = np.random.default_rng(73)
    x = rng.uniform(-6, 6, rows)
    y = rng.uniform(-4, 4, rows)
    z = (
        np.sin(x * 1.4) * np.cos(y * 1.8)
        + 0.55 * np.sin((x + y) * 3)
        + 0.18 * rng.normal(size=rows)
    )
    frame = pd.DataFrame({"x": x, "y": y, "z": z})
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=(-6, 6), y_range=(-4, 4)
    )
    agg = canvas.points(frame, "x", "y", agg=ds.mean("z"))
    raster = tf.shade(
        agg,
        cmap=["#192b5c", "#3167a3", "#54d6c6", "#ffd166", "#ff6f91"],
        how="linear",
        min_alpha=160,
    ).to_pil()
    return _decorate(
        raster,
        title="Streaming terrain scan",
        subtitle=(
            "Irregular samples aggregate into a mean-elevation raster without "
            "interpolation fiction."
        ),
        rows=rows,
        legend=[("low", "#192b5c"), ("mid", "#54d6c6"), ("high", "#ff6f91")],
    )


def _geo_routes(rows: int) -> Image.Image:
    steps = 72
    paths = max(120, rows // steps)
    pid = np.repeat(np.arange(paths), steps)
    t = np.tile(np.linspace(0, 1, steps), paths)
    a = (pid * 37) % 360 * np.pi / 180
    b = (pid * 83 + 47) % 360 * np.pi / 180
    x0 = np.cos(a) * 5.5
    y0 = np.sin(a) * 3.3
    x1 = np.cos(b) * 5.5
    y1 = np.sin(b) * 3.3
    lift = np.sin(t * np.pi) * (1 + (pid % 11) * 0.09)
    x = x0 * (1 - t) + x1 * t
    y = y0 * (1 - t) + y1 * t + lift
    x = np.column_stack((x.reshape(paths, steps), np.full(paths, np.nan))).ravel()
    y = np.column_stack((y.reshape(paths, steps), np.full(paths, np.nan))).ravel()
    frame = pd.DataFrame({"x": x, "y": y})
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=(-6.5, 6.5), y_range=(-4.4, 5.2)
    )
    agg = canvas.line(frame, "x", "y", agg=ds.count(), axis=0)
    raster = tf.shade(
        agg,
        cmap=["#162b55", "#315f9d", "#54d6c6", "#ffd166", "#ff6f91"],
        how="eq_hist",
        min_alpha=14,
    ).to_pil()
    return _decorate(
        raster,
        title="Global route convergence",
        subtitle="Curved origin-destination traces reveal shared corridors and regional gateways.",
        rows=len(frame),
        legend=[("route", "#315f9d"), ("corridor", "#54d6c6"), ("gateway", "#ff6f91")],
    )


def _rare_outliers(rows: int) -> Image.Image:
    rng = np.random.default_rng(99)
    x = rng.normal(size=rows) * 2.2
    y = 0.6 * x + rng.normal(size=rows) * 1.2
    rare = rng.random(rows) < 0.004
    y[rare] += rng.choice([-1, 1], rare.sum()) * (4 + rng.random(rare.sum()) * 2)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "kind": pd.Categorical(
                np.where(rare, "rare", "baseline"), categories=["baseline", "rare"]
            ),
        }
    )
    canvas = ds.Canvas(
        plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, x_range=(-8, 8), y_range=(-8, 8)
    )
    agg = canvas.points(frame, "x", "y", agg=ds.count_cat("kind"))
    colors = {"baseline": "#54d6c6", "rare": "#ff6f91"}
    raster = tf.shade(agg, color_key=colors, how="eq_hist", min_alpha=24)
    raster = tf.spread(raster, px=2).to_pil()
    return _decorate(
        raster,
        title="Rare-event outlier field",
        subtitle=(
            "A categorical aggregate preserves exceptional observations inside "
            "a dense baseline cloud."
        ),
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
    "phase-portrait": Demo(
        "phase-portrait",
        "Phase-space convergence atlas",
        "Trajectory density",
        420_000,
        _phase_portrait,
    ),
    "fractal-basin": Demo(
        "fractal-basin",
        "Newton fractal basins",
        "Categorical iterative basin",
        500_000,
        _fractal_basin,
    ),
    "parameter-sweep": Demo(
        "parameter-sweep",
        "Logistic-map parameter sweep",
        "Dense bifurcation diagram",
        650_000,
        _parameter_sweep,
    ),
    "event-raster": Demo(
        "event-raster",
        "Distributed event raster",
        "Categorical event aggregation",
        1_200_000,
        _event_raster,
    ),
    "uncertainty-fan": Demo(
        "uncertainty-fan",
        "Monte Carlo uncertainty fan",
        "Ensemble line density",
        480_000,
        _uncertainty_fan,
    ),
    "network-density": Demo(
        "network-density",
        "Edge-density network field",
        "Connection density",
        520_000,
        _network_density,
    ),
    "terrain-scan": Demo(
        "terrain-scan",
        "Streaming terrain scan",
        "Mean scalar aggregation",
        1_000_000,
        _terrain_scan,
    ),
    "geo-routes": Demo(
        "geo-routes", "Global route convergence", "Route density", 580_000, _geo_routes
    ),
    "rare-outliers": Demo(
        "rare-outliers",
        "Rare-event outlier field",
        "Categorical anomaly density",
        1_100_000,
        _rare_outliers,
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
