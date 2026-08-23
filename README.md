# datashader-demos

Three production-shaped transparent PNG renderers for data that is too large to draw mark-by-mark: clustered point density, trajectory bundles, and categorical mixing.

```bash
uv sync --group dev
uv run datashader-demos render
uv run datashader-demos validate
uv run pytest
```

The primary artifacts are `out/*-transparent.png`. Each has a manifest with row count, render time, dimensions, and alpha/content pixel statistics. All datasets are deterministic synthetic fixtures.

## Transparent PNG gallery

| Point density | Trajectory bundle | Category mixing |
|---|---|---|
| ![Large point density](out/point-density-transparent.png) | ![Trajectory density](out/trajectory-bundle-transparent.png) | ![Categorical mixing](out/category-mix-transparent.png) |
