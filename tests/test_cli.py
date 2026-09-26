"""CLI surface tests (ADR-1168 version visibility)."""

from importlib.metadata import version as dist_version

import pytest

from datashader_demos.cli import parser


def test_version_flag_prints_manifest_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"datashader-demos {dist_version('datashader-demos')}\n"
