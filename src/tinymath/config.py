"""Small, strict YAML configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """A configuration and the source path it was loaded from."""

    values: Mapping[str, Any]
    path: Path

    def require(self, key: str, expected_type: type | tuple[type, ...] | None = None) -> Any:
        if key not in self.values:
            raise KeyError(f"Missing required configuration key: {key}")
        value = self.values[key]
        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeError(
                f"Configuration key {key!r} must be {expected_type}, got {type(value).__name__}"
            )
        return value


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML mapping and reject malformed or empty configs."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return ExperimentConfig(values=values, path=config_path)
