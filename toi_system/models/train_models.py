from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from ..utils.config import TOIPaths, TrainingConfig
from ..utils.data_utils import prepare_dataset, split_dataset


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, cfg: TrainingConfig) -> tuple[RandomForestClassifier, StandardScaler]:
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X_train)

	model = RandomForestClassifier(
		n_estimators=cfg.rf_estimators,
		max_depth=cfg.rf_max_depth,
		min_samples_split=cfg.rf_min_samples_split,
		min_samples_leaf=cfg.rf_min_samples_leaf,
		random_state=cfg.random_state,
		n_jobs=-1,
	)
	model.fit(X_scaled, y_train)
	return model, scaler


def build_tf_model(input_dim: int) -> tf.keras.Model:
	model = tf.keras.Sequential(
		[
			tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
			tf.keras.layers.Dropout(0.3),
			tf.keras.layers.Dense(64, activation="relu"),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(32, activation="relu"),
			tf.keras.layers.Dense(1, activation="sigmoid"),
		]
	)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		loss="binary_crossentropy",
		metrics=["accuracy"],
	)
	return model


def train_tensorflow(
	X_train: np.ndarray,
	y_train: np.ndarray,
	cfg: TrainingConfig,
) -> tuple[tf.keras.Model, StandardScaler]:
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X_train)

	model = build_tf_model(X_scaled.shape[1])
	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor="val_accuracy",
			patience=15,
			restore_best_weights=True,
		)
	]

	model.fit(
		X_scaled,
		y_train,
		epochs=120,
		batch_size=32,
		validation_split=cfg.validation_split,
		callbacks=callbacks,
		verbose=0,
	)
	return model, scaler


def evaluate_soft_voting(
	rf_model: RandomForestClassifier,
	rf_scaler: StandardScaler,
	tf_model: tf.keras.Model,
	tf_scaler: StandardScaler,
	X_test: np.ndarray,
	y_test: np.ndarray,
) -> Dict[str, float]:
	X_rf = rf_scaler.transform(X_test)
	X_tf = tf_scaler.transform(X_test)

	rf_pred = rf_model.predict(X_rf)
	tf_pred = (tf_model.predict(X_tf, verbose=0).flatten() > 0.5).astype(int)

	rf_acc = accuracy_score(y_test, rf_pred)
	tf_acc = accuracy_score(y_test, tf_pred)

	ensemble_proba = 0.75 * rf_model.predict_proba(X_rf)[:, 1] + 0.25 * tf_model.predict(X_tf, verbose=0).flatten()
	ensemble_pred = (ensemble_proba > 0.5).astype(int)
	director_acc = accuracy_score(y_test, ensemble_pred)

	return {
		"rf_accuracy": rf_acc,
		"tf_accuracy": tf_acc,
		"director_accuracy": director_acc,
	}


def save_models(
	rf_model: RandomForestClassifier,
	rf_scaler: StandardScaler,
	tf_model: tf.keras.Model,
	tf_scaler: StandardScaler,
	output_dir: Path | None = None,
) -> None:
	output = output_dir or TOIPaths.MODEL_DIR
	output.mkdir(parents=True, exist_ok=True)

	joblib.dump(rf_model, output / "rf_model.pkl")
	joblib.dump(rf_scaler, output / "rf_scaler.pkl")
	tf_model.save(output / "tf_model.h5")
	joblib.dump(tf_scaler, output / "tf_scaler.pkl")


def train_all(path: str | Path | None = None) -> Dict[str, float]:
	cfg = TrainingConfig()
	X, y, _, meta = prepare_dataset(path)
	X_train, X_test, y_train, y_test = split_dataset(X, y, cfg)

	rf_model, rf_scaler = train_random_forest(X_train, y_train, cfg)
	tf_model, tf_scaler = train_tensorflow(X_train, y_train, cfg)

	metrics = evaluate_soft_voting(rf_model, rf_scaler, tf_model, tf_scaler, X_test, y_test)
	metrics.update(meta)

	save_models(rf_model, rf_scaler, tf_model, tf_scaler)
	return metrics
