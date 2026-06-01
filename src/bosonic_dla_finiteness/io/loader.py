from pathlib import Path

import yaml

from .models import SystemConfig


def load_from_yaml(path: str | Path) -> SystemConfig:
    """
    Load, validate, and convert a YAML input file.

    Parameters
    ----------
    path : path to the YAML file

    Returns
    -------
    SystemConfig
        The validated system configuration

    Raises
    ------
    pydantic.ValidationError  if any field fails validation
    FileNotFoundError         if the file does not exist
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    # Validate with Pydantic — raises ValidationError with full detail on failure
    validated = SystemConfig(**raw)
    return validated
