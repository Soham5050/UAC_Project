"""Leakage-safe aggregate forecasting and bottleneck-risk models for the UAC dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HORIZON = 7
FEATURE_SOURCE_COLUMNS = [
    "cbp_inflow",
    "cbp_stock",
    "cbp_to_hhs",
    "hhs_stock",
    "hhs_discharged",
    "system_net_flow",
]
FORECAST_TARGETS = ("hhs_discharged", "system_net_flow")
RISK_TARGETS = ("cbp_risk", "hhs_risk")


@dataclass(frozen=True)
class SupervisedFrame:
    features: pd.DataFrame
    targets: dict[str, pd.DataFrame]
    risk_targets: pd.DataFrame
    dates: pd.DataFrame
    history_features: pd.DataFrame
    latest_features: pd.DataFrame


@dataclass(frozen=True)
class TemporalSplits:
    train_index: pd.Index
    validation_index: pd.Index
    test_index: pd.Index


@dataclass
class ForecastResult:
    model_name: str
    validation_mae: float
    test_mae: float
    test_rmse: float
    values: pd.DataFrame


@dataclass
class RiskResult:
    label: str
    available: bool
    probability: float | None
    metrics: dict[str, float | None]
    reason: str | None
    feature_importance: pd.DataFrame


@dataclass
class MLSuiteResult:
    forecasts: dict[str, ForecastResult]
    comparison: pd.DataFrame
    risks: dict[str, RiskResult]
    lstm_available: bool
    lstm_reason: str | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _require_columns(frame: pd.DataFrame) -> None:
    required = {
        "date",
        *FEATURE_SOURCE_COLUMNS,
        "transfer_efficiency_ratio",
        "discharge_effectiveness",
        "cbp_bottleneck",
        "hhs_bottleneck",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing ML input columns: {', '.join(missing)}")


def build_feature_frame(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Return historical-only predictors, indexed by reporting-session row."""
    _require_columns(enriched_df)
    data = enriched_df.sort_values("date").reset_index(drop=True)
    values: dict[str, pd.Series] = {}
    for column in FEATURE_SOURCE_COLUMNS:
        for lag in (1, 2, 3, 7, 14):
            values[f"{column}_lag_{lag}"] = data[column].shift(lag)
        for window in (7, 14):
            values[f"{column}_mean_{window}"] = data[column].rolling(window).mean().shift(1)
            values[f"{column}_std_{window}"] = data[column].rolling(window).std().shift(1)
    for column in ("transfer_efficiency_ratio", "discharge_effectiveness", "cbp_bottleneck", "hhs_bottleneck"):
        values[f"{column}_lag_1"] = data[column].shift(1).astype(float)
    weekday = data["date"].dt.dayofweek
    month = data["date"].dt.month
    values["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    values["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    values["month_sin"] = np.sin(2 * np.pi * month / 12)
    values["month_cos"] = np.cos(2 * np.pi * month / 12)
    return pd.DataFrame(values, index=data.index)


def build_supervised_frame(enriched_df: pd.DataFrame, horizon: int = HORIZON) -> SupervisedFrame:
    """Pair historical features with only future targets, without filling absent sessions."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one reporting session.")
    data = enriched_df.sort_values("date").reset_index(drop=True)
    all_features = build_feature_frame(data)
    targets = {
        target: pd.DataFrame({f"horizon_{step}": data[target].shift(-step) for step in range(1, horizon + 1)})
        for target in FORECAST_TARGETS
    }
    dates = pd.DataFrame({f"horizon_{step}": data["date"].shift(-step) for step in range(1, horizon + 1)})
    risk_targets = pd.DataFrame(
        {
            "cbp_risk": pd.concat([data["cbp_bottleneck"].shift(-step) for step in range(1, horizon + 1)], axis=1).any(axis=1),
            "hhs_risk": pd.concat([data["hhs_bottleneck"].shift(-step) for step in range(1, horizon + 1)], axis=1).any(axis=1),
        }
    )
    eligible = all_features.notna().all(axis=1) & dates.notna().all(axis=1)
    history_features = all_features.dropna().copy()
    if history_features.empty or not eligible.any():
        raise ValueError("Not enough history to construct ML features and future targets.")
    return SupervisedFrame(
        features=all_features.loc[eligible].copy(),
        targets={name: target.loc[eligible].copy() for name, target in targets.items()},
        risk_targets=risk_targets.loc[eligible].copy(),
        dates=dates.loc[eligible].copy(),
        history_features=history_features,
        latest_features=history_features.tail(1),
    )


def make_temporal_splits(frame: SupervisedFrame, test_size: int = 90) -> TemporalSplits:
    index = frame.features.index
    if len(index) < 120:
        raise ValueError("At least 120 eligible reporting sessions are required for ML evaluation.")
    test_length = min(test_size, max(30, len(index) // 4))
    development = index[:-test_length]
    validation_length = max(24, len(development) // 5)
    if len(development) <= validation_length + 30:
        raise ValueError("Not enough chronological observations for train, validation, and test periods.")
    return TemporalSplits(development[:-validation_length], development[-validation_length:], index[-test_length:])


def _regressor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=140,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def _interval_frame(predictions: np.ndarray, residuals: np.ndarray) -> pd.DataFrame:
    low, high = np.quantile(residuals, [0.10, 0.90])
    return pd.DataFrame(
        {
            "session": [f"Session +{step}" for step in range(1, len(predictions) + 1)],
            "prediction": predictions,
            "lower_80": predictions + low,
            "upper_80": predictions + high,
        }
    )


def _forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    return float(mean_absolute_error(actual.ravel(), predicted.ravel())), float(
        mean_squared_error(actual.ravel(), predicted.ravel()) ** 0.5
    )


def _fit_direct_forecast(frame: SupervisedFrame, splits: TemporalSplits, target_name: str) -> ForecastResult:
    validation_actual, validation_predicted, test_actual, test_predicted, future = [], [], [], [], []
    for horizon in frame.targets[target_name].columns:
        target = frame.targets[target_name][horizon]
        model = _regressor()
        model.fit(frame.features.loc[splits.train_index], target.loc[splits.train_index])
        validation_predicted.append(model.predict(frame.features.loc[splits.validation_index]))
        test_predicted.append(model.predict(frame.features.loc[splits.test_index]))
        validation_actual.append(target.loc[splits.validation_index].to_numpy())
        test_actual.append(target.loc[splits.test_index].to_numpy())
        final_model = _regressor()
        final_model.fit(frame.features, target)
        future.append(float(final_model.predict(frame.latest_features)[0]))
    validation_error = np.concatenate(validation_actual) - np.concatenate(validation_predicted)
    test_mae, test_rmse = _forecast_metrics(np.concatenate(test_actual), np.concatenate(test_predicted))
    return ForecastResult(
        "Tabular ML",
        float(np.abs(validation_error).mean()),
        test_mae,
        test_rmse,
        _interval_frame(np.asarray(future), validation_error),
    )


def _persistence_forecast(frame: SupervisedFrame, splits: TemporalSplits, target_name: str) -> ForecastResult:
    column = f"{target_name}_lag_1"
    validation_base = frame.features.loc[splits.validation_index, column].to_numpy()
    test_base = frame.features.loc[splits.test_index, column].to_numpy()
    horizon_count = len(frame.targets[target_name].columns)
    validation_actual = frame.targets[target_name].loc[splits.validation_index].to_numpy()
    test_actual = frame.targets[target_name].loc[splits.test_index].to_numpy()
    validation_predicted = np.repeat(validation_base[:, None], horizon_count, axis=1)
    test_predicted = np.repeat(test_base[:, None], horizon_count, axis=1)
    validation_error = validation_actual - validation_predicted
    test_mae, test_rmse = _forecast_metrics(test_actual, test_predicted)
    latest = float(frame.latest_features.iloc[0][column])
    return ForecastResult(
        "Persistence baseline",
        float(np.abs(validation_error).mean()),
        test_mae,
        test_rmse,
        _interval_frame(np.repeat(latest, horizon_count), validation_error.ravel()),
    )


def _sequence_arrays(
    frame: SupervisedFrame,
    indices: pd.Index,
    target_name: str,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    window: int = 21,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    history = frame.history_features
    scaled_history = feature_scaler.transform(history)
    position = {int(row_index): position for position, row_index in enumerate(history.index)}
    sequences, targets = [], []
    for row_index in indices:
        current = position.get(int(row_index))
        if current is None or current < window - 1:
            continue
        sequences.append(scaled_history[current - window + 1 : current + 1])
        targets.append(frame.targets[target_name].loc[row_index].to_numpy())
    if not sequences:
        return None, None
    return np.asarray(sequences, dtype=np.float32), target_scaler.transform(np.asarray(targets, dtype=float)).astype(np.float32)


def _fit_lstm_forecast(
    frame: SupervisedFrame,
    splits: TemporalSplits,
    target_name: str,
    epochs: int,
    seed: int,
) -> ForecastResult | None:
    try:
        import torch
        from torch import nn
    except ImportError:
        return None
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    feature_scaler = StandardScaler().fit(frame.features.loc[splits.train_index])
    target_scaler = StandardScaler().fit(frame.targets[target_name].loc[splits.train_index])
    train_x, train_y = _sequence_arrays(frame, splits.train_index, target_name, feature_scaler, target_scaler)
    valid_x, valid_y = _sequence_arrays(frame, splits.validation_index, target_name, feature_scaler, target_scaler)
    test_x, test_y = _sequence_arrays(frame, splits.test_index, target_name, feature_scaler, target_scaler)
    if any(value is None for value in (train_x, train_y, valid_x, valid_y, test_x, test_y)):
        return None

    class LSTMForecaster(nn.Module):
        def __init__(self, inputs: int, outputs: int):
            super().__init__()
            self.lstm = nn.LSTM(inputs, 20, batch_first=True)
            self.dropout = nn.Dropout(0.10)
            self.output = nn.Linear(20, outputs)

        def forward(self, values):
            return self.output(self.dropout(self.lstm(values)[0][:, -1, :]))

    def train_model(x_train, y_train, x_valid, y_valid, total_epochs: int):
        model = LSTMForecaster(x_train.shape[2], y_train.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
        loss_fn = nn.MSELoss()
        best_state, best_loss, stale = None, float("inf"), 0
        for _ in range(total_epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(torch.from_numpy(x_train)), torch.from_numpy(y_train))
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_fn(model(torch.from_numpy(x_valid)), torch.from_numpy(y_valid)))
            if validation_loss < best_loss - 1e-6:
                best_state, best_loss, stale = copy.deepcopy(model.state_dict()), validation_loss, 0
            else:
                stale += 1
                if stale >= 6:
                    break
        model.load_state_dict(best_state)
        return model

    model = train_model(train_x, train_y, valid_x, valid_y, epochs)
    model.eval()
    with torch.no_grad():
        validation_predicted = target_scaler.inverse_transform(model(torch.from_numpy(valid_x)).numpy())
        test_predicted = target_scaler.inverse_transform(model(torch.from_numpy(test_x)).numpy())
    validation_actual = target_scaler.inverse_transform(valid_y)
    test_actual = target_scaler.inverse_transform(test_y)
    validation_error = validation_actual - validation_predicted
    test_mae, test_rmse = _forecast_metrics(test_actual, test_predicted)

    all_scaler = StandardScaler().fit(frame.history_features)
    all_target_scaler = StandardScaler().fit(frame.targets[target_name])
    all_x, all_y = _sequence_arrays(frame, frame.features.index, target_name, all_scaler, all_target_scaler)
    latest_history = all_scaler.transform(frame.history_features.tail(21)).astype(np.float32)[None, :, :]
    final_model = train_model(all_x, all_y, all_x, all_y, max(4, epochs // 2))
    final_model.eval()
    with torch.no_grad():
        future = all_target_scaler.inverse_transform(final_model(torch.from_numpy(latest_history)).numpy())[0]
    return ForecastResult(
        "LSTM",
        float(np.abs(validation_error).mean()),
        test_mae,
        test_rmse,
        _interval_frame(future, validation_error.ravel()),
    )


def _fit_risk_models(frame: SupervisedFrame, splits: TemporalSplits) -> dict[str, RiskResult]:
    results: dict[str, RiskResult] = {}
    for label in RISK_TARGETS:
        train_target = frame.risk_targets.loc[splits.train_index, label].astype(int)
        if train_target.nunique() < 2:
            results[label] = RiskResult(label, False, None, {}, "Risk model unavailable: training data contains one class.", pd.DataFrame(columns=["feature", "importance"]))
            continue
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        learning_rate=0.05,
                        max_leaf_nodes=12,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(frame.features.loc[splits.train_index], train_target)
        test_target = frame.risk_targets.loc[splits.test_index, label].astype(int)
        test_probability = model.predict_proba(frame.features.loc[splits.test_index])[:, 1]
        prediction = (test_probability >= 0.60).astype(int)
        metrics: dict[str, float | None] = {
            "precision": float(precision_score(test_target, prediction, zero_division=0)),
            "recall": float(recall_score(test_target, prediction, zero_division=0)),
            "f1": float(f1_score(test_target, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_target, test_probability)) if test_target.nunique() == 2 else None,
            "pr_auc": float(average_precision_score(test_target, test_probability)) if test_target.nunique() == 2 else None,
        }
        if test_target.nunique() == 2:
            importance = permutation_importance(
                model,
                frame.features.loc[splits.test_index],
                test_target,
                scoring="average_precision",
                n_repeats=4,
                random_state=42,
            )
            importance_table = pd.DataFrame(
                {"feature": frame.features.columns, "importance": importance.importances_mean}
            ).sort_values("importance", ascending=False)
        else:
            importance_table = pd.DataFrame(columns=["feature", "importance"])
        final_model = copy.deepcopy(model).fit(frame.features, frame.risk_targets[label].astype(int))
        probability = float(final_model.predict_proba(frame.latest_features)[0, 1])
        results[label] = RiskResult(label, True, probability, metrics, None, importance_table)
    return results


def run_ml_pipeline(enriched_df: pd.DataFrame, lstm_epochs: int = 20) -> MLSuiteResult:
    """Fit/evaluate models chronologically and return a current seven-session outlook."""
    frame = build_supervised_frame(enriched_df)
    splits = make_temporal_splits(frame)
    tabular = {target: _fit_direct_forecast(frame, splits, target) for target in FORECAST_TARGETS}
    baseline = {target: _persistence_forecast(frame, splits, target) for target in FORECAST_TARGETS}
    lstm = {target: _fit_lstm_forecast(frame, splits, target, lstm_epochs, 42) for target in FORECAST_TARGETS}
    lstm_available = all(result is not None for result in lstm.values())
    forecasts: dict[str, ForecastResult] = {}
    rows = []
    for target in FORECAST_TARGETS:
        candidates = [baseline[target], tabular[target]] + ([lstm[target]] if lstm_available else [])
        active = min(candidates, key=lambda result: result.validation_mae)
        forecasts[target] = active
        for result in [baseline[target], tabular[target], *( [lstm[target]] if lstm_available else [] )]:
            rows.append(
                {
                    "Target": target.replace("_", " ").title(),
                    "Model": result.model_name,
                    "Validation MAE": result.validation_mae,
                    "Test MAE": result.test_mae,
                    "Test RMSE": result.test_rmse,
                    "Active": result.model_name == active.model_name,
                }
            )
    test_dates = frame.dates.loc[splits.test_index, "horizon_1"]
    return MLSuiteResult(
        forecasts=forecasts,
        comparison=pd.DataFrame(rows),
        risks=_fit_risk_models(frame, splits),
        lstm_available=lstm_available,
        lstm_reason=None if lstm_available else "LSTM benchmark was unavailable because a chronological segment was too short or PyTorch could not run.",
        test_start=test_dates.min(),
        test_end=test_dates.max(),
    )
