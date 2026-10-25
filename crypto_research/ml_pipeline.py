"""Causal hourly direction-prediction experiments and walk-forward evaluation.

The target is the sign of the close-to-close return over the next four hourly
bars. It is a direction label, not a claim that a trade would be profitable.
Trading returns are evaluated separately at next-hour open with modeled costs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
import ccxt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crypto_research.backtest import (
    BacktestConfig,
    DeterministicRuleProvider,
    run_backtest,
    validate_backtest_candles,
)
from crypto_research.config import Settings, load_settings
from crypto_research.evaluation import _fingerprint_dataset
from crypto_research.exceptions import ConfigurationError, InsufficientHistoryError, StorageError
from crypto_research.market_data import create_public_exchange, fetch_ohlcv
from crypto_research.risk import RiskLimits
from crypto_research.storage import init_db


HORIZON_BARS = 4
TIMEFRAME = "1h"
TIMEFRAME_SECONDS = 3_600
PROBABILITY_ENTRY_THRESHOLD = 0.55
RANDOM_SEED = 42
FEATURE_COLUMNS = (
    "return_1",
    "return_4",
    "return_12",
    "return_24",
    "return_48",
    "return_168",
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",
    "sma_24_gap_pct",
    "sma_50_gap_pct",
    "sma_200_gap_pct",
    "atr_14_pct",
    "realized_vol_24",
    "realized_vol_168",
    "volume_change_1",
    "volume_zscore_24",
    "trend_50_200_pct",
)
MODEL_SPECS = (
    ("logistic_regression", "scaled logistic regression baseline"),
    ("hist_gradient_boosting", "histogram gradient-boosted trees"),
)


@dataclass(frozen=True)
class WalkForwardFold:
    train: pd.Index
    validation: pd.Index
    fold_number: int


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.Index
    validation: pd.Index
    out_of_sample: pd.Index
    boundaries: dict[str, str]
    walk_forward_folds: tuple[WalkForwardFold, ...]


def _code_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_feature_dataset(
    candles: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str = TIMEFRAME,
    horizon_bars: int = HORIZON_BARS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Validate OHLCV and build strictly causal features and forward labels."""
    if timeframe != TIMEFRAME:
        raise ConfigurationError("The initial ML pipeline supports hourly candles only.")
    if horizon_bars != HORIZON_BARS:
        raise ConfigurationError("The initial prediction target is fixed at four hourly bars.")
    frame = validate_backtest_candles(candles, symbol, TIMEFRAME_SECONDS)
    if len(frame) < 1_000:
        raise InsufficientHistoryError(
            f"Hourly training requires at least 1000 valid candles; received {len(frame)}."
        )

    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    returns = close.pct_change()
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    losses = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rs = gains / losses.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((losses == 0) & (gains > 0), 100.0)
    rsi = rsi.mask((losses == 0) & (gains == 0), 50.0)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - close.shift(1)).abs(),
            (frame["low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    volume_mean = volume.rolling(24, min_periods=24).mean()
    volume_std = volume.rolling(24, min_periods=24).std(ddof=0)

    features = pd.DataFrame(index=frame.index)
    for horizon in (1, 4, 12, 24, 48, 168):
        features[f"return_{horizon}"] = close.pct_change(horizon)
    features["rsi_14"] = rsi / 100.0
    features["macd_pct"] = macd / close
    features["macd_signal_pct"] = macd_signal / close
    features["macd_hist_pct"] = (macd - macd_signal) / close
    for window in (24, 50, 200):
        sma = close.rolling(window, min_periods=window).mean()
        features[f"sma_{window}_gap_pct"] = close / sma - 1.0
    features["atr_14_pct"] = atr / close
    features["realized_vol_24"] = returns.rolling(24, min_periods=24).std(ddof=0) * math.sqrt(24)
    features["realized_vol_168"] = returns.rolling(168, min_periods=168).std(ddof=0) * math.sqrt(168)
    features["volume_change_1"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    # A constant-volume window is valid market data with no relative volume signal;
    # encode that neutral condition as zero instead of discarding every such row.
    features["volume_zscore_24"] = ((volume - volume_mean) / volume_std.replace(0.0, np.nan)).fillna(0.0)
    sma_50 = close.rolling(50, min_periods=50).mean()
    sma_200 = close.rolling(200, min_periods=200).mean()
    features["trend_50_200_pct"] = sma_50 / sma_200 - 1.0

    forward_return = close.shift(-horizon_bars) / close - 1.0
    target = (forward_return > 0.0).astype("int8")
    valid = features.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    valid &= forward_return.notna()
    valid_timestamps = pd.DatetimeIndex(frame.loc[valid, "timestamp"])
    features = features.loc[valid, list(FEATURE_COLUMNS)].astype(float)
    target = target.loc[valid].astype("int8")
    forward_return = forward_return.loc[valid].astype(float)
    features.index = valid_timestamps
    target.index = valid_timestamps
    forward_return.index = valid_timestamps
    if len(features) < 600:
        raise InsufficientHistoryError(
            f"Feature warm-up and four-bar target leave only {len(features)} samples; need at least 600."
        )
    return features, target, forward_return, frame


def chronological_split(index: pd.Index, *, horizon_bars: int = HORIZON_BARS) -> ChronologicalSplit:
    """60/20/20 ordered split, purging labels that cross split boundaries."""
    n = len(index)
    first = int(n * 0.60)
    second = int(n * 0.80)
    train_end = first - horizon_bars
    validation_end = second - horizon_bars
    train = index[:train_end]
    validation = index[first:validation_end]
    out_of_sample = index[second:]
    if min(len(train), len(validation), len(out_of_sample)) < 100:
        raise InsufficientHistoryError(
            "Chronological split needs at least 100 observations in each train, validation, and holdout block."
        )
    folds: list[WalkForwardFold] = []
    for fold_number, (start_ratio, end_ratio) in enumerate(
        ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80)), start=1
    ):
        fold_start = int(n * start_ratio)
        fold_end = int(n * end_ratio) - horizon_bars
        fold_train = index[:fold_start - horizon_bars]
        fold_validation = index[fold_start:fold_end]
        if min(len(fold_train), len(fold_validation)) < 100:
            raise InsufficientHistoryError(
                "Walk-forward validation needs at least 100 observations per fold."
            )
        folds.append(
            WalkForwardFold(
                train=fold_train,
                validation=fold_validation,
                fold_number=fold_number,
            )
        )
    return ChronologicalSplit(
        train=train,
        validation=validation,
        out_of_sample=out_of_sample,
        boundaries={
            "train_start_utc": str(index[0]),
            "train_end_utc": str(train[-1]),
            "validation_start_utc": str(validation[0]),
            "validation_end_utc": str(validation[-1]),
            "out_of_sample_start_utc": str(out_of_sample[0]),
            "out_of_sample_end_utc": str(out_of_sample[-1]),
            "purge_bars": str(horizon_bars),
        },
        walk_forward_folds=tuple(folds),
    )


def _make_model(model_name: str) -> Any:
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1_000,
                        random_state=RANDOM_SEED,
                        class_weight=None,
                    ),
                ),
            ]
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        )
    raise ConfigurationError(f"Unsupported model name: {model_name}")


def _classification_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype("int8")
    truth = y_true.to_numpy(dtype="int8")
    brier = float(np.mean((probabilities - truth) ** 2))
    try:
        loss = float(log_loss(truth, probabilities, labels=[0, 1]))
    except ValueError:
        loss = None
    bins = []
    for lower in np.linspace(0.0, 0.8, 5):
        upper = lower + 0.2
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if mask.any():
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(mask.sum()),
                    "mean_predicted_probability": float(probabilities[mask].mean()),
                    "observed_up_rate": float(truth[mask].mean()),
                }
            )
    return {
        "sample_count": int(len(truth)),
        "positive_rate": float(truth.mean()) if len(truth) else 0.0,
        "brier_score": brier,
        "log_loss": loss,
        "precision_at_0_5": float(precision_score(truth, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(truth, predicted, zero_division=0)),
        "probability_calibration_bins": bins,
    }


def _drawdown(equity: list[float]) -> float:
    peak = equity[0] if equity else 0.0
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd * 100.0


def _strategy_metrics(
    probabilities: np.ndarray,
    feature_index: pd.Index,
    candles: pd.DataFrame,
    *,
    fee_rate: float,
    slippage_pct: float,
    allocation_fraction: float,
    entry_threshold: float,
    hold_bars: int = HORIZON_BARS,
    initial_equity: float = 10_000.0,
) -> dict[str, Any]:
    timestamp_to_row = {timestamp: i for i, timestamp in enumerate(candles["timestamp"])}
    equity = initial_equity
    curve = [equity]
    net_trade_pnls: list[float] = []
    fees_total = 0.0
    slippage_total = 0.0
    entries = 0
    windows = 0
    turnover_usd = 0.0
    for offset in range(0, len(feature_index), hold_bars):
        signal_ts = feature_index[offset]
        row_idx = timestamp_to_row.get(signal_ts)
        if row_idx is None or row_idx + hold_bars + 1 >= len(candles):
            continue
        windows += 1
        probability = float(probabilities[offset])
        pnl = 0.0
        if probability >= entry_threshold:
            entry_raw = float(candles.iloc[row_idx + 1]["open"])
            exit_raw = float(candles.iloc[row_idx + hold_bars + 1]["open"])
            entry_fill = entry_raw * (1.0 + slippage_pct)
            exit_fill = exit_raw * (1.0 - slippage_pct)
            allocation = equity * allocation_fraction
            turnover_usd += allocation * 2.0
            quantity = allocation / (entry_fill * (1.0 + fee_rate))
            buy_fee = quantity * entry_fill * fee_rate
            sell_fee = quantity * exit_fill * fee_rate
            pnl = quantity * (exit_fill - entry_fill) - buy_fee - sell_fee
            fees_total += buy_fee + sell_fee
            slippage_total += quantity * (entry_raw * slippage_pct + exit_raw * slippage_pct)
            entries += 1
            net_trade_pnls.append(pnl)
            equity += pnl
        curve.append(equity)

    interval_returns = [
        (right - left) / left for left, right in zip(curve, curve[1:]) if left > 0
    ]
    if len(interval_returns) > 1:
        deviation = float(np.std(interval_returns, ddof=1))
        sharpe = (
            math.sqrt((365.0 * 24.0) / hold_bars)
            * float(np.mean(interval_returns))
            / deviation
            if deviation > 1e-12
            else 0.0
        )
    else:
        sharpe = 0.0
    gross_wins = sum(pnl for pnl in net_trade_pnls if pnl > 0)
    gross_losses = abs(sum(pnl for pnl in net_trade_pnls if pnl < 0))
    return {
        "initial_equity_usd": initial_equity,
        "final_equity_usd": equity,
        "net_return_pct": (equity / initial_equity - 1.0) * 100.0,
        "max_drawdown_pct": _drawdown(curve),
        "sharpe_ratio_annualized": sharpe,
        "sharpe_frequency": "non-overlapping four-hour intervals",
        "sharpe_annualization_factor": math.sqrt((365.0 * 24.0) / hold_bars),
        "profit_factor": gross_wins / gross_losses if gross_losses else (float("inf") if gross_wins else 0.0),
        "trades": entries,
        "turnover_usd": turnover_usd,
        "turnover_multiple": turnover_usd / initial_equity,
        "trading_fees_usd": fees_total,
        "modeled_slippage_usd": slippage_total,
        "total_trading_costs_usd": fees_total + slippage_total,
        "average_exposure_pct": (entries / windows * allocation_fraction * 100.0) if windows else 0.0,
        "entry_probability_threshold": entry_threshold,
        "allocation_fraction_per_trade": allocation_fraction,
        "no_trade_baseline_return_pct": 0.0,
        "benchmark_cost_assumptions": "Model positions include two fees and adverse slippage per round trip; no-trade benchmark costs are zero.",
    }


def _buy_and_hold_metrics(
    feature_index: pd.Index,
    candles: pd.DataFrame,
    *,
    fee_rate: float,
    slippage_pct: float,
    initial_equity: float = 10_000.0,
) -> dict[str, Any]:
    timestamp_to_row = {timestamp: i for i, timestamp in enumerate(candles["timestamp"])}
    if len(feature_index) == 0:
        raise InsufficientHistoryError("Cannot calculate buy-and-hold benchmark without test candles.")
    start = timestamp_to_row[pd.Timestamp(feature_index[0])]
    end = min(
        timestamp_to_row[pd.Timestamp(feature_index[-1])] + HORIZON_BARS + 1,
        len(candles) - 1,
    )
    if start + 1 >= len(candles) or end >= len(candles):
        raise InsufficientHistoryError("Insufficient trailing candles for the test benchmark window.")
    entry_raw = float(candles.iloc[start + 1]["open"])
    exit_raw = float(candles.iloc[end]["open"])
    entry_fill = entry_raw * (1.0 + slippage_pct)
    exit_fill = exit_raw * (1.0 - slippage_pct)
    quantity = initial_equity / (entry_fill * (1.0 + fee_rate))
    final_equity = quantity * exit_fill * (1.0 - fee_rate)
    entry_fee = quantity * entry_fill * fee_rate
    exit_fee = quantity * exit_fill * fee_rate
    slippage_cost = quantity * (entry_raw * slippage_pct + exit_raw * slippage_pct)
    path = [initial_equity]
    for row_idx in range(start + 1, end):
        marked_value = quantity * float(candles.iloc[row_idx]["close"])
        path.append(marked_value + (initial_equity - quantity * entry_fill - entry_fee))
    path.append(final_equity)
    interval_returns = [
        right / left - 1.0 for left, right in zip(path, path[1:]) if left > 0
    ]
    std = float(np.std(interval_returns, ddof=1)) if len(interval_returns) > 1 else 0.0
    sharpe = (
        math.sqrt(365.0 * 24.0) * float(np.mean(interval_returns)) / std
        if std > 1e-12 else 0.0
    )
    return {
        "name": "btc_buy_and_hold",
        "initial_equity_usd": initial_equity,
        "final_equity_usd": final_equity,
        "net_return_pct": (final_equity / initial_equity - 1.0) * 100.0,
        "max_drawdown_pct": _drawdown(path),
        "sharpe_ratio_annualized": sharpe,
        "sharpe_frequency": "hourly marks, annualized with sqrt(365*24)",
        "trading_fees_usd": entry_fee + exit_fee,
        "modeled_slippage_usd": slippage_cost,
        "total_trading_costs_usd": entry_fee + exit_fee + slippage_cost,
        "assumption": "100% buy-and-hold on test window, with one modeled entry and exit cost; not exposure-matched.",
    }


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def train_models_on_candles(
    candles: pd.DataFrame,
    *,
    symbol: str,
    data_source: str,
    db_path: str | Path,
    artifact_root: str | Path = "artifacts",
    risk_limits: RiskLimits | None = None,
    fee_rate: float = 0.001,
    slippage_pct: float = 0.0005,
) -> list[dict[str, Any]]:
    """Train baseline and gradient-boosted models, persist artifacts and scores."""
    features, target, forward_return, validated = build_feature_dataset(
        candles, symbol=symbol
    )
    split = chronological_split(features.index)
    dataset_hash = _fingerprint_dataset(validated)
    artifact_base = Path(artifact_root)
    symbol_slug = symbol.replace("/", "").replace("-", "").lower()
    dataset_path = artifact_base / "datasets" / f"ohlcv_{symbol_slug}_{dataset_hash}.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        validated.to_csv(dataset_path, index=False)
    db = init_db(db_path)
    limits = risk_limits or RiskLimits()
    now = datetime.now(timezone.utc).isoformat()
    code_revision = _code_revision()
    source_fp = _source_fingerprint()
    config = {
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "target": "binary sign of close[t+4]/close[t]-1; 1=up, 0=flat-or-down",
        "target_horizon_bars": HORIZON_BARS,
        "target_candle_policy": "Features use the fully closed hourly candle at t; labels use closes through t+4.",
        "entry_execution": "A signal at close t may enter at open t+1 and exit at open t+5.",
        "entry_probability_threshold": PROBABILITY_ENTRY_THRESHOLD,
        "allocation_fraction_per_trade": limits.max_portfolio_allocation_pct,
        "fee_rate_per_side": fee_rate,
        "slippage_pct_per_side": slippage_pct,
        "risk_limits": asdict(limits),
        "feature_columns": list(FEATURE_COLUMNS),
        "split_ratios": {"train": 0.60, "validation": 0.20, "out_of_sample": 0.20},
        "purge_bars_each_split": HORIZON_BARS,
        "random_seed": RANDOM_SEED,
        "models": [dict(name=name, description=description) for name, description in MODEL_SPECS],
        "model_hyperparameters": {
            "logistic_regression": {
                "C": 1.0, "max_iter": 1_000, "class_weight": None,
                "imputation": "median fit within each training fold", "scaling": "standard fit within each training fold",
            },
            "hist_gradient_boosting": {
                "max_iter": 150, "max_leaf_nodes": 15,
                "learning_rate": 0.08, "l2_regularization": 1.0,
            },
        },
        "walk_forward_folds": [
            {"train_end_utc": str(fold.train[-1]), "validation_start_utc": str(fold.validation[0]),
             "validation_end_utc": str(fold.validation[-1])}
            for fold in split.walk_forward_folds
        ],
        "data_source": data_source,
        "dataset_sha256": dataset_hash,
        "dataset_start_utc": validated["timestamp"].iloc[0].isoformat(),
        "dataset_end_utc": validated["timestamp"].iloc[-1].isoformat(),
        "code_revision": code_revision,
        "source_fingerprint_sha256": source_fp,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "ccxt_version": ccxt.__version__,
        "joblib_version": joblib.__version__,
    }
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        for model_name, description in MODEL_SPECS:
            batch_id = uuid.uuid4().hex[:10]
            run_id = f"ml_{batch_id}"
            model_version = f"{model_name}_{dataset_hash[:10]}_{batch_id}"
            model_config = {**config, "model_name": model_name, "model_description": description}
            config_sha = hashlib.sha256(
                json.dumps(model_config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            model_path = artifact_base / "models" / model_version
            created_at = datetime.now(timezone.utc).isoformat()
            db.execute(
                """INSERT INTO model_runs (
                       model_version, run_id, model_name, symbol, timeframe, status,
                       dataset_sha256, data_start_utc, data_end_utc, artifact_path,
                       metadata_json, metrics_json, created_at_utc
                   ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, '{}', ?)""",
                (
                    model_version, run_id, model_name, symbol, TIMEFRAME,
                    dataset_hash, config["dataset_start_utc"], config["dataset_end_utc"],
                    str(model_path), json.dumps(_serialize(model_config), sort_keys=True), created_at,
                ),
            )
            db.commit()
            try:
                walk_forward_scores: list[dict[str, Any]] = []
                for fold in split.walk_forward_folds:
                    fold_target = target.loc[fold.train]
                    if fold_target.nunique() < 2:
                        raise InsufficientHistoryError(
                            f"{model_name} walk-forward fold {fold.fold_number} has only one target class."
                        )
                    fold_model = _make_model(model_name)
                    fold_model.fit(features.loc[fold.train], fold_target)
                    fold_probabilities = fold_model.predict_proba(
                        features.loc[fold.validation]
                    )[:, 1]
                    walk_forward_scores.append(
                        {
                            "fold": fold.fold_number,
                            "train_start_utc": str(fold.train[0]),
                            "train_end_utc": str(fold.train[-1]),
                            "validation_start_utc": str(fold.validation[0]),
                            "validation_end_utc": str(fold.validation[-1]),
                            "metrics": _classification_metrics(
                                target.loc[fold.validation], fold_probabilities
                            ),
                        }
                    )

                if target.loc[split.train].nunique() < 2:
                    raise InsufficientHistoryError(
                        f"{model_name} training split contains only one target class."
                    )
                model = _make_model(model_name)
                model.fit(features.loc[split.train], target.loc[split.train])
                probabilities: dict[str, np.ndarray] = {}
                classification_scores: dict[str, Any] = {}
                for split_name, split_index in (
                    ("validation", split.validation),
                    ("out_of_sample", split.out_of_sample),
                ):
                    proba = model.predict_proba(features.loc[split_index])[:, 1]
                    probabilities[split_name] = proba
                    classification_scores[split_name] = _classification_metrics(
                        target.loc[split_index], proba
                    )

                validation_strategy = _strategy_metrics(
                    probabilities["validation"], split.validation, validated,
                    fee_rate=fee_rate,
                    slippage_pct=slippage_pct,
                    allocation_fraction=limits.max_portfolio_allocation_pct,
                    entry_threshold=PROBABILITY_ENTRY_THRESHOLD,
                )
                out_strategy = _strategy_metrics(
                    probabilities["out_of_sample"], split.out_of_sample, validated,
                    fee_rate=fee_rate,
                    slippage_pct=slippage_pct,
                    allocation_fraction=limits.max_portfolio_allocation_pct,
                    entry_threshold=PROBABILITY_ENTRY_THRESHOLD,
                )
                rule_start = pd.Timestamp(split.out_of_sample[0]).isoformat()
                rule_end = pd.Timestamp(split.out_of_sample[-1]).isoformat()
                rule_report = run_backtest(
                    BacktestConfig(
                        run_id=f"baseline_{batch_id}",
                        symbols=[symbol],
                        timeframe=TIMEFRAME,
                        initial_cash_usd=10_000.0,
                        fee_rate=fee_rate,
                        slippage_pct=slippage_pct,
                        start_date=rule_start,
                        end_date=rule_end,
                        risk_limits=limits,
                        sqlite_db_path=":memory:",
                    ),
                    candles_by_symbol={symbol: validated},
                    decision_provider=DeterministicRuleProvider(),
                )
                rule_baseline = {
                    "net_return_pct": rule_report.total_return_pct,
                    "max_drawdown_pct": rule_report.max_drawdown_pct,
                    "sharpe_ratio": "not annualized; backtester reports daily-compatible convention only",
                    "trades": rule_report.completed_trades,
                    "fees_paid_usd": rule_report.total_fees_paid_usd,
                }
                metrics = {
                    "walk_forward_validation": walk_forward_scores,
                    "classification": classification_scores,
                    "model_strategy_validation": validation_strategy,
                    "model_strategy_out_of_sample": out_strategy,
                    "benchmarks_out_of_sample": {
                        "buy_and_hold": _buy_and_hold_metrics(
                            split.out_of_sample,
                            validated,
                            fee_rate=fee_rate,
                            slippage_pct=slippage_pct,
                        ),
                        "no_trade": {
                            "name": "cash_only",
                            "net_return_pct": 0.0,
                            "max_drawdown_pct": 0.0,
                            "trading_costs_usd": 0.0,
                        },
                    },
                    "simple_rule_baseline_out_of_sample": rule_baseline,
                    "out_of_sample_guard": "Do not select or tune models using this split; reserve for final comparison.",
                }

                model_path.mkdir(parents=True, exist_ok=False)
                joblib.dump(model, model_path / "model.joblib")
                metadata = {
                    **model_config,
                    "model_version": model_version,
                    "run_id": run_id,
                    "experiment_config_sha256": config_sha,
                    "training_started_at_utc": created_at,
                    "training_completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "feature_count": len(FEATURE_COLUMNS),
                    "split_boundaries": split.boundaries,
                    "fit_sample_count": len(split.train),
                    "validation_sample_count": len(split.validation),
                    "out_of_sample_sample_count": len(split.out_of_sample),
                    "target_direction_counts": {
                        str(int(k)): int(v) for k, v in target.loc[split.train].value_counts().items()
                    },
                    "metrics": _serialize(metrics),
                }
                (model_path / "metadata.json").write_text(
                    json.dumps(_serialize(metadata), indent=2, allow_nan=False),
                    encoding="utf-8",
                )
                pd.DataFrame(
                    {
                        "as_of_utc": split.out_of_sample.astype(str),
                        "probability_up": probabilities["out_of_sample"],
                        "predicted_direction": (probabilities["out_of_sample"] >= 0.5).astype(int),
                        "observed_direction": target.loc[split.out_of_sample].to_numpy(),
                        "forward_return": forward_return.loc[split.out_of_sample].to_numpy(),
                    }
                ).to_csv(model_path / "out_of_sample_predictions.csv", index=False)

                db.executemany(
                    """INSERT INTO model_predictions (
                           model_version, split_name, as_of_utc, symbol, probability_up,
                           predicted_direction, observed_direction, forward_return, created_at_utc
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            model_version,
                            split_name,
                            pd.Timestamp(ts).isoformat(),
                            symbol,
                            float(prob),
                            int(prob >= 0.5),
                            int(target.loc[ts]),
                            float(forward_return.loc[ts]),
                            datetime.now(timezone.utc).isoformat(),
                        )
                        for split_name in ("validation", "out_of_sample")
                        for ts, prob in zip(
                            split.validation if split_name == "validation" else split.out_of_sample,
                            probabilities[split_name],
                        )
                    ],
                )
                db.execute(
                    """UPDATE model_runs SET status='completed', metrics_json=?, completed_at_utc=?
                       WHERE model_version=?""",
                    (
                        json.dumps(_serialize(metrics), sort_keys=True, allow_nan=False),
                        datetime.now(timezone.utc).isoformat(),
                        model_version,
                    ),
                )
                db.commit()
                results.append(
                    {
                        "model_version": model_version,
                        "model_name": model_name,
                        "run_id": run_id,
                        "artifact_path": str(model_path),
                        "metrics": _serialize(metrics),
                        "metadata": metadata,
                    }
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                db.execute(
                    "UPDATE model_runs SET status='failed', error_message=?, completed_at_utc=? WHERE model_version=?",
                    (error[:2_000], datetime.now(timezone.utc).isoformat(), model_version),
                )
                db.commit()
                failures.append(f"{model_name}: {error}")
    finally:
        db.close()
    if failures:
        raise StorageError(
            "One or more models failed; failures are recorded and no failed run was marked complete: "
            + "; ".join(failures)
        )
    return results


def fetch_and_train(
    symbol: str,
    *,
    candle_limit: int = 10_000,
    timeframe: str = TIMEFRAME,
    db_path: str | Path = "crypto_research.db",
    artifact_root: str | Path = "artifacts",
) -> list[dict[str, Any]]:
    """Fetch public closed candles then run the deterministic training pipeline."""
    if symbol not in {"BTC/USDT", "ETH/USDT"}:
        raise ConfigurationError("Initial model training supports BTC/USDT and ETH/USDT only.")
    if timeframe != TIMEFRAME:
        raise ConfigurationError("Initial model training supports timeframe='1h' only.")
    if candle_limit < 1_000 or candle_limit > 50_000:
        raise ConfigurationError("Candle limit must be between 1000 and 50000.")
    settings = load_settings()
    exchange = create_public_exchange(settings.exchange_id)
    fetch_settings = Settings(
        exchange_id=settings.exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        candle_limit=candle_limit,
        openai_model=settings.openai_model,
        openai_api_key_configured=False,
    )
    candles = fetch_ohlcv(fetch_settings, exchange=exchange)
    endpoint = getattr(exchange, "base_url", "public OHLCV API")
    return train_models_on_candles(
        candles,
        symbol=symbol,
        data_source=f"Public {settings.exchange_id} OHLCV ({endpoint})",
        db_path=db_path,
        artifact_root=artifact_root,
        risk_limits=settings.to_risk_limits(),
        fee_rate=settings.paper_fee_rate,
        slippage_pct=settings.paper_slippage_pct,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and walk-forward evaluate hourly research models.")
    parser.add_argument("--symbols", nargs="+", choices=["BTC/USDT", "ETH/USDT"], default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--candle-limit", type=int, default=10_000)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args(argv)
    settings = load_settings()
    db_path = args.db_path or settings.sqlite_db_path
    for symbol in args.symbols:
        print(f"Fetching and training hourly models for {symbol}...")
        results = fetch_and_train(
            symbol,
            candle_limit=args.candle_limit,
            db_path=db_path,
            artifact_root=args.artifact_root,
        )
        for result in results:
            test = result["metrics"]["classification"]["out_of_sample"]
            strategy = result["metrics"]["model_strategy_out_of_sample"]
            print(
                f"{result['model_name']} {result['model_version']}: "
                f"Brier={test['brier_score']:.4f}, log-loss={test['log_loss']:.4f}, "
                f"net OOS return={strategy['net_return_pct']:+.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
