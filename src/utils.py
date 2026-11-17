"""Shared helpers: project paths and configuration loading."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    """Read the YAML configuration file and return it as a dictionary."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(relative_path):
    """Resolve a path from the configuration against the project root."""
    return PROJECT_ROOT / relative_path
