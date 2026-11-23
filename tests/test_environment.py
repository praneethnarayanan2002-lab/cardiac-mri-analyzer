"""Checks that the environment matches what pyproject.toml promises."""

import sys
import tomllib
from pathlib import Path

import cardiac

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_runs_inside_project_venv() -> None:
    project_venv = PYPROJECT.parent / ".venv"
    assert Path(sys.prefix).resolve() == project_venv.resolve()


def test_python_version_is_311() -> None:
    assert sys.version_info[:2] == (3, 11)


def test_package_version_matches_pyproject() -> None:
    with PYPROJECT.open("rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert cardiac.__version__ == declared


def test_every_runtime_dependency_is_pinned_exactly() -> None:
    with PYPROJECT.open("rb") as f:
        project = tomllib.load(f)
    groups = project["dependency-groups"]
    specs = project["project"]["dependencies"] + groups["app"] + groups["dev"]
    loose = [spec for spec in specs if "==" not in spec]
    assert loose == [], f"unpinned dependencies: {loose}"
