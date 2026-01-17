"""
Quantum Trading AI - ML Training Pipeline
==========================================
Trains ensemble models using walk-forward cross-validation.

Model Architecture:
- LightGBM (fast, handles large datasets)
- XGBoost (accurate, regularization)
- Meta-learner (calibrated logistic regression)

Key Features:
- Walk-forward CV to prevent look-ahead bias
- Purged gap between train/validation
- Probability calibration for confidence scores
- Model versioning and persistence
"""
import logging
import os
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, classification_report, confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb

from .config import settings


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Walk-forward settings
    n_splits: int = 5
    train_days: int = 180
    val_days: int = 30
    purge_days: int = 1  # Gap between train/val to prevent leakage

    # LightGBM parameters
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    })

    # Early stopping
    early_stopping_rounds: int = 50


@dataclass
class TrainingResult:
    """Results from model training"""
    model_version: str
    train_date: datetime
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    model_paths: Dict[str, str]


class ModelTrainer:
    """
    Trains ensemble models for trading signal prediction.

    Uses walk-forward cross-validation to simulate real-world conditions
    where we only train on past data.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.models: Dict[str, Any] = {}
        self.meta_model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.model_version: str = ""

    def _generate_version(self, symbol: str, timeframe: str) -> str:
        """Generate unique model version string"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"ensemble_{symbol}_{timeframe}_{timestamp}"

    def _walk_forward_split(
        self,
        n_samples: int,
        candles_per_day: int = 1440
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward cross-validation splits.

        Each split has:
        - Training set: train_days of data
        - Purge gap: purge_days of data (excluded)
        - Validation set: val_days of data

        Returns list of (train_indices, val_indices) tuples.
        """
        splits = []

        train_size = self.config.train_days * candles_per_day
        val_size = self.config.val_days * candles_per_day
        purge_size = self.config.purge_days * candles_per_day
        step_size = val_size

        # Start from enough data for first training
        current_end = train_size + purge_size + val_size

        while current_end <= n_samples and len(splits) < self.config.n_splits:
            val_end = current_end
            val_start = val_end - val_size
            train_end = val_start - purge_size
            train_start = max(0, train_end - train_size)

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)

            splits.append((train_idx, val_idx))
            current_end += step_size

        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        symbol: str,
        timeframe: str,
    ) -> TrainingResult:
        """
        Train the ensemble model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,) with values 0, 1, 2
            feature_names: List of feature names
            symbol: Trading pair symbol
            timeframe: Signal timeframe

        Returns:
            TrainingResult with metrics and model paths
        """
        logger.info(f"Training model for {symbol} {timeframe}")
        logger.info(f"Data shape: {X.shape}, Classes: {np.unique(y)}")

        self.feature_names = feature_names
        self.model_version = self._generate_version(symbol, timeframe)

        # Get walk-forward splits
        splits = self._walk_forward_split(len(X))

        if len(splits) == 0:
            raise ValueError("Not enough data for walk-forward CV")

        # Collect predictions from all folds
        all_val_indices = []
        all_val_preds_lgbm = []
        all_val_preds_xgb = []
        all_val_true = []

        # Train on each fold
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Training fold {fold + 1}/{len(splits)}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train LightGBM
            lgbm_model = lgb.LGBMClassifier(**self.config.lgbm_params)
            lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),  # Suppress output
                ]
            )

            # Train XGBoost
            xgb_model = xgb.XGBClassifier(**self.config.xgb_params)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Get validation predictions
            lgbm_pred = lgbm_model.predict_proba(X_val)
            xgb_pred = xgb_model.predict_proba(X_val)

            all_val_indices.extend(val_idx)
            all_val_preds_lgbm.append(lgbm_pred)
            all_val_preds_xgb.append(xgb_pred)
            all_val_true.extend(y_val)

        # Combine predictions
        all_val_preds_lgbm = np.vstack(all_val_preds_lgbm)
        all_val_preds_xgb = np.vstack(all_val_preds_xgb)
        all_val_true = np.array(all_val_true)

        # Stack predictions for meta-learner
        meta_features = np.hstack([all_val_preds_lgbm, all_val_preds_xgb])

        # Train meta-learner with calibration
        logger.info("Training meta-learner...")
        base_meta = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            random_state=42
        )
        self.meta_model = CalibratedClassifierCV(base_meta, cv=3, method="isotonic")
        self.meta_model.fit(meta_features, all_val_true)

        # Final predictions
        final_probs = self.meta_model.predict_proba(meta_features)
        final_preds = final_probs.argmax(axis=1)

        # Calculate metrics
        metrics = self._calculate_metrics(all_val_true, final_preds, final_probs)

        # Train final models on all data
        logger.info("Training final models on full dataset...")
        self.models["lgbm"] = lgb.LGBMClassifier(**self.config.lgbm_params)
        self.models["lgbm"].fit(X, y)

        self.models["xgb"] = xgb.XGBClassifier(**self.config.xgb_params)
        self.models["xgb"].fit(X, y)

        # Feature importance (average of both models)
        feature_importance = self._get_feature_importance()

        # Save models
        model_paths = self._save_models(symbol, timeframe)

        # Classification report
        class_names = ["SHORT", "NEUTRAL", "LONG"]
        report = classification_report(
            all_val_true, final_preds,
            target_names=class_names
        )

        # Confusion matrix
        cm = confusion_matrix(all_val_true, final_preds)

        logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")

        return TrainingResult(
            model_version=self.model_version,
            train_date=datetime.now(timezone.utc),
            metrics=metrics,
            feature_importance=feature_importance,
            confusion_matrix=cm,
            classification_report=report,
            model_paths=model_paths,
        )

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "log_loss": log_loss(y_true, y_prob),
        }

        # Per-class metrics
        for i, name in enumerate(["short", "neutral", "long"]):
            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)

            metrics[f"precision_{name}"] = precision_score(binary_true, binary_pred, zero_division=0)
            metrics[f"recall_{name}"] = recall_score(binary_true, binary_pred, zero_division=0)
            metrics[f"f1_{name}"] = f1_score(binary_true, binary_pred, zero_division=0)

        # Win rate calculation (for trading signals)
        # We consider LONG signals correct if y_true == 2
        # And SHORT signals correct if y_true == 0
        long_signals = y_pred == 2
        short_signals = y_pred == 0

        if long_signals.sum() > 0:
            metrics["win_rate_long"] = (y_true[long_signals] == 2).mean()
        else:
            metrics["win_rate_long"] = 0.0

        if short_signals.sum() > 0:
            metrics["win_rate_short"] = (y_true[short_signals] == 0).mean()
        else:
            metrics["win_rate_short"] = 0.0

        # Combined win rate (LONG + SHORT signals)
        all_signals = long_signals | short_signals
        if all_signals.sum() > 0:
            correct = ((y_pred[all_signals] == 2) & (y_true[all_signals] == 2)) | \
                      ((y_pred[all_signals] == 0) & (y_true[all_signals] == 0))
            metrics["win_rate_combined"] = correct.mean()
        else:
            metrics["win_rate_combined"] = 0.0

        return metrics

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get averaged feature importance from ensemble"""
        importance = {}

        # LightGBM importance
        lgbm_imp = self.models["lgbm"].feature_importances_

        # XGBoost importance
        xgb_imp = self.models["xgb"].feature_importances_

        # Average
        avg_imp = (lgbm_imp + xgb_imp) / 2

        for i, name in enumerate(self.feature_names):
            importance[name] = float(avg_imp[i])

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def _save_models(self, symbol: str, timeframe: str) -> Dict[str, str]:
        """Save trained models to disk"""
        model_dir = settings.model_dir
        os.makedirs(model_dir, exist_ok=True)

        paths = {}

        # Save LightGBM
        lgbm_path = os.path.join(model_dir, f"{self.model_version}_lgbm.joblib")
        joblib.dump(self.models["lgbm"], lgbm_path)
        paths["lgbm"] = lgbm_path

        # Save XGBoost
        xgb_path = os.path.join(model_dir, f"{self.model_version}_xgb.joblib")
        joblib.dump(self.models["xgb"], xgb_path)
        paths["xgb"] = xgb_path

        # Save meta-model
        meta_path = os.path.join(model_dir, f"{self.model_version}_meta.joblib")
        joblib.dump(self.meta_model, meta_path)
        paths["meta"] = meta_path

        # Save feature names
        features_path = os.path.join(model_dir, f"{self.model_version}_features.joblib")
        joblib.dump(self.feature_names, features_path)
        paths["features"] = features_path

        logger.info(f"Models saved to {model_dir}")
        return paths

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the ensemble.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (predictions, probabilities)
            - predictions: Array of class labels (0, 1, 2)
            - probabilities: Array of shape (n_samples, 3) with class probabilities
        """
        if not self.models or self.meta_model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions from base models
        lgbm_probs = self.models["lgbm"].predict_proba(X)
        xgb_probs = self.models["xgb"].predict_proba(X)

        # Stack for meta-learner
        meta_features = np.hstack([lgbm_probs, xgb_probs])

        # Final predictions
        probabilities = self.meta_model.predict_proba(meta_features)
        predictions = probabilities.argmax(axis=1)

        return predictions, probabilities

    def load(self, model_version: str, model_dir: Optional[str] = None) -> None:
        """Load a trained model from disk"""
        if model_dir is None:
            model_dir = settings.model_dir

        self.model_version = model_version

        # Load models
        self.models["lgbm"] = joblib.load(
            os.path.join(model_dir, f"{model_version}_lgbm.joblib")
        )
        self.models["xgb"] = joblib.load(
            os.path.join(model_dir, f"{model_version}_xgb.joblib")
        )
        self.meta_model = joblib.load(
            os.path.join(model_dir, f"{model_version}_meta.joblib")
        )
        self.feature_names = joblib.load(
            os.path.join(model_dir, f"{model_version}_features.joblib")
        )

        logger.info(f"Loaded model {model_version}")
