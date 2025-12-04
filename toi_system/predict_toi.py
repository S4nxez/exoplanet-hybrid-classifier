from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from toi_system.core.director import TOIDirector
from toi_system.utils.config import TOIPaths
from toi_system.utils.data_utils import build_feature_matrix, load_raw_dataset


def _load_models(model_dir: Path):
	rf_model = joblib.load(model_dir / "rf_model.pkl")
	rf_scaler = joblib.load(model_dir / "rf_scaler.pkl")
	tf_model = tf.keras.models.load_model(model_dir / "tf_model.h5")
	tf_scaler = joblib.load(model_dir / "tf_scaler.pkl")
	return rf_model, rf_scaler, tf_model, tf_scaler


def run_inference(limit: int | None = None, dataset_path: str | Path | None = None):
	df = load_raw_dataset(dataset_path)
	features, feature_names = build_feature_matrix(df)
	if limit:
		features = features[:limit]

	director = TOIDirector()
	rf_model, rf_scaler, tf_model, tf_scaler = _load_models(TOIPaths.MODEL_DIR)
	director.configure(rf_model, rf_scaler, tf_model, tf_scaler)

	preds, models = director.predict(features, return_model_choice=True)
	summary = {
		"samples": int(len(preds)),
		"positives": int(preds.sum()),
		"negatives": int((preds == 0).sum()),
		"model_usage": director.get_stats(),
	}

	return preds, models, summary, feature_names


def main():
	parser = argparse.ArgumentParser(description="Run TOI director predictions.")
	parser.add_argument("--dataset", type=str, default=str(TOIPaths.DATASET), help="Path to a CSV with TOI samples")
	parser.add_argument("--limit", type=int, default=20, help="Number of samples to score")
	parser.add_argument("--json", action="store_true", help="Output JSON summary only")
	args = parser.parse_args()

	preds, models, summary, _ = run_inference(args.limit, args.dataset)

	if args.json:
		print(json.dumps(summary, indent=2))
		return

	print("TOI prediction summary")
	print(json.dumps(summary, indent=2))
	print("First predictions:")
	for idx, (pred, model) in enumerate(zip(preds[:10], models[:10])):
		print(f"Sample {idx:02d}: class={pred} via {model}")


if __name__ == "__main__":
	main()
