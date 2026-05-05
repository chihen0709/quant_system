"""PyTorch training and inference for the first hybrid DL signal."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - exercised when torch is not installed.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


DEFAULT_SEQUENCE_LENGTH = 120
DEFAULT_HORIZON = 20
DEFAULT_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA5",
    "MA20",
    "MA50",
    "MA150",
    "MA200",
    "RSI",
    "MACD_Osc",
    "trend_score",
    "vcp_score",
    "bb_width",
    "bb_width_pctile",
    "bb_breakout",
]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for train_dl_model/predict_dl_signals. "
            "Install dependencies with `pip install -r requirements.txt` or use Docker after rebuilding."
        )


class CNNBiLSTMAttention(nn.Module if nn is not None else object):
    """A compact CNN + BiLSTM + attention model for daily OHLCV sequences."""

    def __init__(self, input_size: int, hidden_size: int = 48, output_size: int = 3, dropout: float = 0.30):
        _require_torch()
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2)
        self.activation = nn.GELU()
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_dropout(self.activation(self.conv(x)).transpose(1, 2))
        x, _ = self.lstm(x)
        weights = torch.softmax(self.attention(x).squeeze(-1), dim=1).unsqueeze(-1)
        context = (x * weights).sum(dim=1)
        return self.head(context)


class TemporalTransformer(nn.Module if nn is not None else object):
    """A small Transformer encoder fallback for sequence experiments."""

    def __init__(self, input_size: int, hidden_size: int = 48, output_size: int = 3, dropout: float = 0.30):
        _require_torch()
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])


@dataclass
class SequenceBundle:
    x: np.ndarray
    y: np.ndarray
    dates: list
    feature_columns: list[str]
    source_ids: np.ndarray


def _future_rolling(series: pd.Series, horizon: int, reducer: str) -> pd.Series:
    shifted = series.shift(-1)
    reversed_series = shifted.iloc[::-1]
    if reducer == "max":
        rolled = reversed_series.rolling(horizon, min_periods=horizon).max()
    elif reducer == "min":
        rolled = reversed_series.rolling(horizon, min_periods=horizon).min()
    else:
        raise ValueError(f"unsupported reducer: {reducer}")
    return rolled.iloc[::-1]


def add_supervised_labels(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    target_return: float = 0.12,
    max_drawdown: float = 0.08,
) -> pd.DataFrame:
    """Add no-lookahead labels for trend, breakout success and risk."""
    out = df.copy()
    future_close = out["Close"].shift(-horizon)
    future_max_close = _future_rolling(out["Close"], horizon, "max")
    future_min_low = _future_rolling(out["Low"], horizon, "min")

    future_return = future_close / out["Close"] - 1.0
    future_best_return = future_max_close / out["Close"] - 1.0
    future_drawdown = future_min_low / out["Close"] - 1.0

    out["label_trend"] = (future_return > 0).astype(float)
    out["label_breakout"] = ((future_best_return >= target_return) & (future_drawdown > -max_drawdown)).astype(float)
    out["label_risk"] = (future_drawdown <= -max_drawdown).astype(float)
    out.loc[future_close.isna(), ["label_trend", "label_breakout", "label_risk"]] = np.nan
    return out


def _build_sequences(
    frames: Iterable[pd.DataFrame],
    feature_columns: list[str],
    sequence_length: int,
    horizon: int,
    target_return: float = 0.12,
    max_drawdown: float = 0.08,
) -> SequenceBundle:
    xs = []
    ys = []
    dates = []
    source_ids = []
    for source_idx, frame in enumerate(frames):
        labeled = add_supervised_labels(
            frame,
            horizon=horizon,
            target_return=target_return,
            max_drawdown=max_drawdown,
        )
        data = labeled.copy()
        for col in feature_columns:
            if col not in data.columns:
                data[col] = 0.0
        data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
        data[feature_columns] = data[feature_columns].ffill().fillna(0.0)

        feature_matrix = data[feature_columns].to_numpy(dtype=np.float32)
        labels = data[["label_trend", "label_breakout", "label_risk"]].to_numpy(dtype=np.float32)

        stop = len(data) - horizon
        for end_idx in range(sequence_length - 1, stop):
            label = labels[end_idx]
            if np.isnan(label).any():
                continue
            window = feature_matrix[end_idx - sequence_length + 1 : end_idx + 1]
            if np.isnan(window).any():
                continue
            xs.append(window)
            ys.append(label)
            dates.append(data.index[end_idx])
            source_ids.append(source_idx)

    if not xs:
        raise ValueError("not enough clean rows to build supervised DL sequences")
    return SequenceBundle(np.stack(xs), np.stack(ys), dates, feature_columns, np.asarray(source_ids, dtype=np.int32))


def _time_split_bundle(bundle: SequenceBundle, validation_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split each source chronologically so every ticker contributes train and validation rows."""
    validation_fraction = float(np.clip(validation_fraction, 0.05, 0.50))
    train_indices = []
    val_indices = []
    for source_id in np.unique(bundle.source_ids):
        indices = np.flatnonzero(bundle.source_ids == source_id)
        if len(indices) < 4:
            train_indices.extend(indices.tolist())
            continue
        split = max(1, int(len(indices) * (1.0 - validation_fraction)))
        split = min(split, len(indices) - 1)
        train_indices.extend(indices[:split].tolist())
        val_indices.extend(indices[split:].tolist())

    if not val_indices:
        split = max(1, int(len(bundle.x) * (1.0 - validation_fraction)))
        split = min(split, len(bundle.x) - 1)
        train_indices = list(range(split))
        val_indices = list(range(split, len(bundle.x)))

    return bundle.x[train_indices], bundle.y[train_indices], bundle.x[val_indices], bundle.y[val_indices]


def _make_model(model_type: str, input_size: int, hidden_size: int = 48, dropout: float = 0.30):
    if model_type == "temporal_transformer":
        return TemporalTransformer(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
    if model_type == "cnn_bilstm_attention":
        return CNNBiLSTMAttention(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
    raise ValueError(f"unsupported model_type: {model_type}")


def _default_tickers(market: str) -> list[str]:
    if market.upper() == "TW":
        return ["2330.TW", "2454.TW", "2303.TW", "2317.TW"]
    if market.upper() == "US":
        return ["AAPL", "NVDA", "MSFT", "AMZN"]
    return ["2330.TW", "2454.TW", "AAPL", "NVDA"]


def train_dl_model(
    market: str,
    train_start: str,
    train_end: str,
    model_type: str = "cnn_bilstm_attention",
    tickers: list[str] | None = None,
    feature_frames: dict[str, pd.DataFrame] | list[pd.DataFrame] | None = None,
    output_dir: str = "models",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    horizon: int = DEFAULT_HORIZON,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-3,
    hidden_size: int = 48,
    dropout: float = 0.30,
    grad_clip: float = 1.0,
    target_return: float = 0.12,
    max_drawdown: float = 0.08,
    random_state: int = 42,
    verbose: bool = True,
    validation_fraction: float = 0.20,
    early_stopping_patience: int = 4,
    min_delta: float = 1e-4,
) -> str:
    """Train the first DL signal model and return the artifact path."""
    _require_torch()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if verbose:
        selected_tickers = tickers or _default_tickers(market)
        print(
            f"[DL] start training market={market} model={model_type} "
            f"range={train_start}..{train_end} tickers={','.join(selected_tickers)}",
            flush=True,
        )

    if feature_frames is None:
        from .features import build_feature_frame

        frames = []
        selected_tickers = tickers or _default_tickers(market)
        for idx, ticker in enumerate(selected_tickers, start=1):
            if verbose:
                print(f"[DL] ({idx}/{len(selected_tickers)}) building feature frame for {ticker}...", flush=True)
            frame = build_feature_frame(ticker, market=None, start=train_start, end=train_end)
            frames.append(frame)
            if verbose:
                print(f"[DL] ({idx}/{len(selected_tickers)}) {ticker}: {len(frame)} bars ready", flush=True)
    elif isinstance(feature_frames, dict):
        frames = list(feature_frames.values())
    else:
        frames = list(feature_frames)

    if verbose:
        print(
            f"[DL] building supervised sequences length={sequence_length} horizon={horizon}...",
            flush=True,
        )
    bundle = _build_sequences(
        frames,
        DEFAULT_FEATURE_COLUMNS,
        sequence_length,
        horizon,
        target_return=target_return,
        max_drawdown=max_drawdown,
    )
    x_train, y_train, x_val, y_val = _time_split_bundle(bundle, validation_fraction)
    if verbose:
        train_rates = y_train.mean(axis=0)
        val_rates = y_val.mean(axis=0)
        print(
            f"[DL] sequences={len(bundle.x)} train={len(x_train)} val={len(x_val)} "
            f"features={len(bundle.feature_columns)}",
            flush=True,
        )
        print(
            "[DL] label positive rates "
            f"train trend={train_rates[0]:.3f} breakout={train_rates[1]:.3f} risk={train_rates[2]:.3f} | "
            f"val trend={val_rates[0]:.3f} breakout={val_rates[1]:.3f} risk={val_rates[2]:.3f}",
            flush=True,
        )

    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(
            f"[DL] device={device} batch_size={batch_size} epochs={epochs} "
            f"hidden_size={hidden_size} dropout={dropout:.2f} lr={learning_rate:g} weight_decay={weight_decay:g}",
            flush=True,
        )
    model = _make_model(
        model_type,
        input_size=len(bundle.feature_columns),
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    pos_weight = np.divide(negatives, np.maximum(positives, 1.0))
    pos_weight = np.clip(pos_weight, 0.25, 8.0).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))
    if verbose:
        print(
            f"[DL] BCE pos_weight trend={pos_weight[0]:.3f} breakout={pos_weight[1]:.3f} risk={pos_weight[2]:.3f}",
            flush=True,
        )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.tensor(x_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    history = []
    best_state = None
    best_val = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
        train_loss /= max(1, len(x_train))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if verbose:
            print(
                f"[DL] epoch {epoch:03d}/{epochs:03d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val={best_val:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} patience={epochs_without_improvement}/{early_stopping_patience}",
                flush=True,
            )
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(
                    f"[DL] early stopping at epoch {epoch}; best validation loss was {best_val:.6f}",
                    flush=True,
                )
            break

    os.makedirs(output_dir, exist_ok=True)
    artifact_path = os.path.join(output_dir, f"{market.lower()}_{model_type}.pt")
    torch.save(
        {
            "model_type": model_type,
            "model_state": best_state or model.state_dict(),
            "feature_columns": bundle.feature_columns,
            "sequence_length": sequence_length,
            "horizon": horizon,
            "scaler": {"mean": mean.tolist(), "std": std.tolist()},
            "history": history,
            "metadata": {
                "market": market,
                "train_start": train_start,
                "train_end": train_end,
                "label_order": ["trend_probability", "breakout_probability", "risk_probability"],
                "validation_fraction": validation_fraction,
                "early_stopping_patience": early_stopping_patience,
                "pos_weight": pos_weight.tolist(),
                "hidden_size": hidden_size,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "target_return": target_return,
                "max_drawdown": max_drawdown,
            },
        },
        artifact_path,
    )

    with open(artifact_path.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "artifact_path": artifact_path,
                "train_sequences": int(len(x_train)),
                "val_sequences": int(len(x_val)),
                "feature_count": int(len(bundle.feature_columns)),
                "validation_fraction": validation_fraction,
                "early_stopping_patience": early_stopping_patience,
            },
            f,
            indent=2,
        )
    if verbose:
        print(f"[DL] saved model to {artifact_path}", flush=True)
    return artifact_path


def predict_dl_signals(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Return dl_signal and probability columns aligned to the input frame index."""
    _require_torch()
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"You are using `torch.load` with `weights_only=False`.*")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_columns = checkpoint["feature_columns"]
    sequence_length = int(checkpoint["sequence_length"])
    mean = np.asarray(checkpoint["scaler"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["scaler"]["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    data = df.copy()
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0.0
    data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    data[feature_columns] = data[feature_columns].ffill().fillna(0.0)
    matrix = ((data[feature_columns].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)

    output = pd.DataFrame(index=data.index)
    output["trend_probability"] = 0.5
    output["breakout_probability"] = 0.5
    output["risk_probability"] = 0.5
    output["dl_signal"] = 0.5

    if len(data) < sequence_length:
        return output

    windows = []
    target_indices = []
    for end_idx in range(sequence_length - 1, len(data)):
        windows.append(matrix[end_idx - sequence_length + 1 : end_idx + 1])
        target_indices.append(data.index[end_idx])

    metadata = checkpoint.get("metadata", {})
    state = checkpoint.get("model_state", {})
    inferred_hidden_size = metadata.get("hidden_size")
    if inferred_hidden_size is None and "conv.weight" in state:
        inferred_hidden_size = int(state["conv.weight"].shape[0])
    model = _make_model(
        checkpoint["model_type"],
        input_size=len(feature_columns),
        hidden_size=int(inferred_hidden_size or 48),
        dropout=float(metadata.get("dropout", 0.30)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(np.stack(windows), dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()

    output.loc[target_indices, "trend_probability"] = probs[:, 0]
    output.loc[target_indices, "breakout_probability"] = probs[:, 1]
    output.loc[target_indices, "risk_probability"] = probs[:, 2]
    output["dl_signal"] = (
        0.30 * output["trend_probability"]
        + 0.50 * output["breakout_probability"]
        + 0.20 * (1.0 - output["risk_probability"])
    ).clip(0.0, 1.0)
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a hybrid DL signal model.")
    parser.add_argument("--market", default="MIXED", choices=["TW", "US", "MIXED"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--model-type", default="cnn_bilstm_attention", choices=["cnn_bilstm_attention", "temporal_transformer"])
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--target-return", type=float, default=0.12)
    parser.add_argument("--max-drawdown", type=float, default=0.08)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    artifact = train_dl_model(
        market=args.market,
        train_start=args.start,
        train_end=end,
        model_type=args.model_type,
        tickers=args.tickers,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        target_return=args.target_return,
        max_drawdown=args.max_drawdown,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
    )
    print(f"DL model saved to {artifact}")


if __name__ == "__main__":
    main()
