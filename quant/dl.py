"""PyTorch training and inference for the first hybrid DL signal."""

from __future__ import annotations

import json
import os
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

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 3):
        _require_torch()
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2)
        self.activation = nn.GELU()
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
            nn.Dropout(0.15),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.activation(self.conv(x)).transpose(1, 2)
        x, _ = self.lstm(x)
        weights = torch.softmax(self.attention(x).squeeze(-1), dim=1).unsqueeze(-1)
        context = (x * weights).sum(dim=1)
        return self.head(context)


class TemporalTransformer(nn.Module if nn is not None else object):
    """A small Transformer encoder fallback for sequence experiments."""

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 3):
        _require_torch()
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=0.15,
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
) -> SequenceBundle:
    xs = []
    ys = []
    dates = []
    for frame in frames:
        labeled = add_supervised_labels(frame, horizon=horizon)
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

    if not xs:
        raise ValueError("not enough clean rows to build supervised DL sequences")
    return SequenceBundle(np.stack(xs), np.stack(ys), dates, feature_columns)


def _make_model(model_type: str, input_size: int):
    if model_type == "temporal_transformer":
        return TemporalTransformer(input_size=input_size)
    if model_type == "cnn_bilstm_attention":
        return CNNBiLSTMAttention(input_size=input_size)
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
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> str:
    """Train the first DL signal model and return the artifact path."""
    _require_torch()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if feature_frames is None:
        from .features import build_feature_frame

        frames = []
        for ticker in tickers or _default_tickers(market):
            frames.append(build_feature_frame(ticker, market=None, start=train_start, end=train_end))
    elif isinstance(feature_frames, dict):
        frames = list(feature_frames.values())
    else:
        frames = list(feature_frames)

    bundle = _build_sequences(frames, DEFAULT_FEATURE_COLUMNS, sequence_length, horizon)
    split = max(1, int(len(bundle.x) * 0.8))
    x_train, y_train = bundle.x[:split], bundle.y[:split]
    x_val, y_val = bundle.x[split:], bundle.y[split:]
    if len(x_val) == 0:
        x_val, y_val = x_train, y_train

    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(model_type, input_size=len(bundle.feature_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

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
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
        train_loss /= max(1, len(x_train))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

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
            },
        },
        artifact_path,
    )

    with open(artifact_path.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "artifact_path": artifact_path}, f, indent=2)
    return artifact_path


def predict_dl_signals(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Return dl_signal and probability columns aligned to the input frame index."""
    _require_torch()
    checkpoint = torch.load(model_path, map_location="cpu")
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

    model = _make_model(checkpoint["model_type"], input_size=len(feature_columns))
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
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    artifact = train_dl_model(
        market=args.market,
        train_start=args.start,
        train_end=end,
        model_type=args.model_type,
        tickers=args.tickers,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )
    print(f"DL model saved to {artifact}")


if __name__ == "__main__":
    main()
