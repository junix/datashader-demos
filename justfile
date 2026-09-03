set shell := ["bash", "-euo", "pipefail", "-c"]

install_bin := home_directory() / "sync" / "bin"

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

# Interpreted CLI launcher (ADR-749).
install: build
    #!/usr/bin/env bash
    mkdir -p "{{ install_bin }}"
    cat > "{{ install_bin }}/datashader-demos" << 'EOF'
    #!/usr/bin/env bash
    cd {{ justfile_directory() }} && uv run datashader-demos "$@"
    EOF
    chmod +x "{{ install_bin }}/datashader-demos"

# Remove generated images.
clean:
    rm -rf out
    mkdir -p out
    touch out/.gitkeep

# Rebuild gallery.html from catalog.json, README.md and the artifacts in out/.
gallery:
    python3 tools/gallery.py
