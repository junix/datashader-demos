set shell := ["zsh", "-cu"]

sync:
    uv sync --group dev

render:
    uv run datashader-demos render

test:
    uv run ruff check .
    uv run pytest

validate:
    uv run datashader-demos validate
