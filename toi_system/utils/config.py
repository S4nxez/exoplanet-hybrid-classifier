from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class TOIPaths:
	"""Centralized filesystem locations for the TOI mission."""

	BASE_DIR = Path(__file__).resolve().parents[1]
	DATA_DIR = BASE_DIR.parent / "data" / "clean"
	DATASET = DATA_DIR / "toi_full.csv"
	MODEL_DIR = BASE_DIR / "saved_models"


class TOIFeatures:
	"""Feature schemas consumed by scalers and directors."""

	CORE = [
		"pl_orbper",
		"pl_trandurh",
		"pl_trandep",
		"pl_rade",
		"pl_insol",
		"pl_eqt",
		"st_tmag",
		"st_dist",
		"st_teff",
		"st_logg",
		"st_rad",
	]

	OPTIONAL = ["toi", "toipfx", "tid", "ctoi_alias"]


class TargetConfig:
	"""Mapping from catalog dispositions to a binary label."""

	COLUMN = "tfopwg_disp"
	POSITIVE = {"CP", "KP", "PC"}
	NEGATIVE = {"APC", "FP", "FA"}
	FILL_VALUE = "FP"


@dataclass(frozen=True)
class TrainingConfig:
	random_state: int = 42
	test_size: float = 0.2
	validation_split: float = 0.2
	rf_estimators: int = 250
	rf_max_depth: int | None = 18
	rf_min_samples_split: int = 4
	rf_min_samples_leaf: int = 2


@dataclass(frozen=True)
class TOIGatingThresholds:
	confidence_gap: float = 0.08
	minimum_confidence: float = 0.12
	complexity_threshold: float = 0.48
