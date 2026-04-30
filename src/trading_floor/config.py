"""Runtime configuration loaded from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    researcher_model: str
    strategist_model: str
    risk_model: str
    trader_model: str
    starting_cash: float
    price_seed: int
    data_dir: Path
    max_position_pct: float
    max_sector_pct: float


def _required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return val


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.getenv("TF_DATA_DIR", "./runs")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        openai_api_key=_required("OPENAI_API_KEY"),
        researcher_model=os.getenv("TF_RESEARCHER_MODEL", "gpt-4o-mini"),
        strategist_model=os.getenv("TF_STRATEGIST_MODEL", "gpt-4o-mini"),
        risk_model=os.getenv("TF_RISK_MODEL", "gpt-4o-mini"),
        trader_model=os.getenv("TF_TRADER_MODEL", "gpt-4o-mini"),
        starting_cash=float(os.getenv("TF_STARTING_CASH", "1000000")),
        price_seed=int(os.getenv("TF_PRICE_SEED", "42")),
        data_dir=data_dir,
        max_position_pct=float(os.getenv("TF_MAX_POSITION_PCT", "20")),
        max_sector_pct=float(os.getenv("TF_MAX_SECTOR_PCT", "40")),
    )
