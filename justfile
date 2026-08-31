set shell := ["bash", "-euo", "pipefail", "-c"]

default: build

# Sync deps and render every demo into out/.
build:
    uv sync --group dev
    uv run datashader-demos render

# Lint, unit tests, then the render/validate gate.
test: build
    uv run ruff check .
    uv run pytest
    uv run datashader-demos validate

# Demos repo — no binary, no launcher (ADR-749: nothing to install).
install:
    @echo "datashader-demos: demos repo, nothing to install"

# Remove generated images.
clean:
    rm -rf out
    mkdir -p out
    touch out/.gitkeep
