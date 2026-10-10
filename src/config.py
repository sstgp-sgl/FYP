"""One configuration file for the Web workflow."""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open() as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    for block, key in [(cfg, "output_dir"), (cfg["segment"], "sam2_checkpoint")]:
        p = Path(block[key])
        block[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg
