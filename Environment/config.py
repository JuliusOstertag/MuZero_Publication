# config.py
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Config:
    episode_length: int = 144
    battery_capacity: float = 100.0

    pv_path: str = "../data/pv.csv"
    wind_path: str = "../data/wind.csv"
    grid_price_path: str = "../data/grid_price.csv"

    # Idee
    reward_weight_import: float = 1.0
    reward_weight_peak: float = 0.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}

        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}

        return cls(**filtered)
