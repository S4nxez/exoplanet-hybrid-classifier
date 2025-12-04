from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import TOIPaths, TOIFeatures, TargetConfig, TrainingConfig


def load_raw_dataset(path: str | Path | None = None) -> pd.DataFrame:
	"""Read the cleaned TOI catalog into a DataFrame."""

	dataset_path = Path(path) if path else TOIPaths.DATASET
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset not found at {dataset_path}")
	return pd.read_csv(dataset_path)


def _require_features(df: pd.DataFrame) -> None:
	missing = [col for col in TOIFeatures.CORE if col not in df.columns]
	if missing:
		raise ValueError(f"Dataset is missing required columns: {missing}")


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
	"""Return an ordered feature matrix aligned with the scalers."""

	_require_features(df)
	matrix = df[TOIFeatures.CORE].copy()
	matrix = matrix.fillna(matrix.median())
	return matrix.to_numpy(dtype=np.float64), TOIFeatures.CORE


def build_target_vector(df: pd.DataFrame) -> np.ndarray:
	"""Map TFOPWG dispositions to a binary label."""

	if TargetConfig.COLUMN not in df.columns:
		raise ValueError(f"Column {TargetConfig.COLUMN} not found in dataset")

	labels = df[TargetConfig.COLUMN].fillna(TargetConfig.FILL_VALUE).astype(str).str.upper()

	positives = labels.isin(TargetConfig.POSITIVE)
	negatives = labels.isin(TargetConfig.NEGATIVE)

	# Ambiguous values default to negative but emit a warning for visibility
	ambiguous = ~(positives | negatives)
	if ambiguous.any():
		unknown_values = labels[ambiguous].unique().tolist()
		print(json.dumps({"warning": "Unknown TOI dispositions", "values": unknown_values}))

	target = np.where(positives, 1, 0)
	target = np.where(negatives, 0, target)
	return target.astype(int)


def prepare_dataset(path: str | Path | None = None) -> Tuple[np.ndarray, np.ndarray, list[str], Dict[str, int]]:
	"""Load, clean, and align features/targets for training."""

	df = load_raw_dataset(path)
	feature_matrix, feature_names = build_feature_matrix(df)
	targets = build_target_vector(df)

	valid_mask = ~np.isnan(feature_matrix).any(axis=1)
	feature_matrix = feature_matrix[valid_mask]
	targets = targets[valid_mask]

	metadata = {
		"samples": int(feature_matrix.shape[0]),
		"features": len(feature_names),
		"positives": int(targets.sum()),
		"negatives": int((targets == 0).sum()),
	}

	return feature_matrix, targets, feature_names, metadata


def split_dataset(
	X: np.ndarray,
	y: np.ndarray,
	config: TrainingConfig | None = None,
):
	cfg = config or TrainingConfig()
	return train_test_split(
		X,
		y,
		test_size=cfg.test_size,
		random_state=cfg.random_state,
		stratify=y,
	)


def fit_scaler(X: np.ndarray) -> StandardScaler:
	scaler = StandardScaler()
	scaler.fit(X)
	return scaler
