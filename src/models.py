"""
Models for the decision-time leakage benchmark.

Implements six model families:
- MomentumModel  : ret_20 score, no training
- RidgeModel     : sklearn Ridge, alpha=1.0
- LightGBMModel  : LGBMRegressor with fixed hyperparameters
- GraphMLPModel  : two-branch PyTorch MLP (base + relation)
- LSTMModel      : single-layer PyTorch LSTM over lookback=60 windows
- TransformerModel: PyTorch Transformer encoder over lookback=60 windows

Public factory
--------------
get_model(model_name) -> BaseModel
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------------
# Optional PyTorch import – private nn.Module classes live inside this block
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True

    class _GraphMLPNet(nn.Module):
        """Two-branch MLP: separate encoders for base and relation features."""

        def __init__(self, base_dim: int, rel_dim: int) -> None:
            super().__init__()
            self.base_branch = nn.Sequential(
                nn.Linear(base_dim, 32), nn.ReLU(),
                nn.Linear(32, 32),       nn.ReLU(),
            )
            self.rel_branch = nn.Sequential(
                nn.Linear(rel_dim, 32), nn.ReLU(),
                nn.Linear(32, 32),      nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x_base: torch.Tensor, x_rel: torch.Tensor) -> torch.Tensor:
            return self.head(
                torch.cat([self.base_branch(x_base), self.rel_branch(x_rel)], dim=-1)
            )

    class _LSTMNet(nn.Module):
        """Single-layer LSTM with a linear output head."""

        def __init__(self, n_features: int, hidden_dim: int = 32) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                n_features, hidden_dim, num_layers=1, dropout=0, batch_first=True
            )
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h, _) = self.lstm(x)        # h: (1, batch, hidden_dim)
            return self.head(h.squeeze(0))  # (batch, 1)

    class _PositionalEncoding(nn.Module):
        """Standard sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10_000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class _TransformerNet(nn.Module):
        """Transformer encoder: linear projection → positional encoding → encoder → head."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 32,
            nhead: int = 4,
            num_encoder_layers: int = 1,
            dim_feedforward: int = 64,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, n_features)
            x = self.pos_enc(self.input_proj(x))  # (batch, seq_len, d_model)
            x = self.encoder(x)                    # (batch, seq_len, d_model)
            return self.head(x[:, -1, :])          # last token → (batch, 1)

except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sequence helper
# ---------------------------------------------------------------------------

def prepare_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a flat feature array into sliding-window sequences.

    Parameters
    ----------
    X:           Feature matrix of shape (n_samples, n_features).
    y:           Target vector of shape (n_samples,).
    lookback:    Number of time steps per sequence window.
    max_samples: If given, keep only the *last* max_samples sequences.

    Returns
    -------
    seqs:   np.ndarray of shape (n, lookback, n_features)
    labels: np.ndarray of shape (n,)
    """
    n = len(X)
    indices = np.arange(lookback - 1, n)
    if max_samples is not None and len(indices) > max_samples:
        indices = indices[-max_samples:]
    seqs = np.stack([X[i - lookback + 1 : i + 1] for i in indices], axis=0)
    labels = y[indices]
    return seqs, labels


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Abstract interface shared by all model families."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the model on training data, optionally using a validation set."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return scalar predictions for each row of X."""

    @abstractmethod
    def reset(self) -> None:
        """Re-initialise all learnable parameters / model state."""


# ---------------------------------------------------------------------------
# 1. Momentum
# ---------------------------------------------------------------------------

class MomentumModel(BaseModel):
    """Momentum baseline: score = ret_20 (first feature column, no training)."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0].copy()

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 2. Ridge
# ---------------------------------------------------------------------------

class RidgeModel(BaseModel):
    """Sklearn Ridge regression with alpha=1.0."""

    def __init__(self) -> None:
        np.random.seed(42)
        self._model = Ridge(alpha=1.0)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def reset(self) -> None:
        np.random.seed(42)
        self._model = Ridge(alpha=1.0)


# ---------------------------------------------------------------------------
# 3. LightGBM
# ---------------------------------------------------------------------------

class LightGBMModel(BaseModel):
    """LightGBM regressor with fixed benchmark hyperparameters."""

    _PARAMS = {
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    }

    def __init__(self) -> None:
        np.random.seed(42)
        self._model = LGBMRegressor(**self._PARAMS)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def reset(self) -> None:
        np.random.seed(42)
        self._model = LGBMRegressor(**self._PARAMS)


# ---------------------------------------------------------------------------
# 4. GraphMLP
# ---------------------------------------------------------------------------

class GraphMLPModel(BaseModel):
    """Two-branch graph MLP.

    The ``fit``/``predict`` interface receives a single matrix X that is the
    horizontal concatenation ``[X_base | X_rel]``.  Internally the model
    splits X at ``base_features_dim`` and routes each part through its own
    encoder branch before merging in the shared head.

    Parameters
    ----------
    base_features_dim:     Number of base (stock-level) features.
    relation_features_dim: Number of relation / graph-neighbour features.
    """

    def __init__(self, base_features_dim: int, relation_features_dim: int) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for GraphMLPModel. "
                "Install it with: pip install torch"
            )
        torch.manual_seed(42)
        np.random.seed(42)
        self._base_dim = base_features_dim
        self._rel_dim = relation_features_dim
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = _GraphMLPNet(base_features_dim, relation_features_dim).to(self._device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train for 1 epoch with batch_size=1024 and Adam lr=1e-3."""
        self._net.train()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        x_base_t = torch.tensor(X_train[:, : self._base_dim], dtype=torch.float32)
        x_rel_t  = torch.tensor(X_train[:, self._base_dim :], dtype=torch.float32)
        y_t      = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(
            TensorDataset(x_base_t, x_rel_t, y_t),
            batch_size=1024,
            shuffle=True,
        )
        for xb, xr, yb in loader:
            xb = xb.to(self._device)
            xr = xr.to(self._device)
            yb = yb.to(self._device)
            optimizer.zero_grad()
            loss_fn(self._net(xb, xr), yb).backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        with torch.no_grad():
            xb = torch.tensor(X[:, : self._base_dim], dtype=torch.float32).to(self._device)
            xr = torch.tensor(X[:, self._base_dim :], dtype=torch.float32).to(self._device)
            return self._net(xb, xr).squeeze(1).cpu().numpy()

    def reset(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self._net = _GraphMLPNet(self._base_dim, self._rel_dim).to(self._device)


# ---------------------------------------------------------------------------
# 5. LSTM
# ---------------------------------------------------------------------------

class LSTMModel(BaseModel):
    """Single-layer LSTM sequence model.

    Converts the flat input matrix to overlapping ``lookback``-step windows
    via :func:`prepare_sequences` before training / inference.

    Parameters
    ----------
    n_features: Number of input features per time step.
    """

    LOOKBACK: int          = 60
    HIDDEN_DIM: int        = 32
    LAYERS: int            = 1
    DROPOUT: float         = 0
    LR: float              = 1e-3
    BATCH_SIZE: int        = 512
    EPOCHS: int            = 1
    MAX_TRAIN_SAMPLES: int = 12_000
    MAX_VAL_SAMPLES: int   = 3_000

    def __init__(self, n_features: int) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTMModel. "
                "Install it with: pip install torch"
            )
        torch.manual_seed(42)
        np.random.seed(42)
        self._n_features = n_features
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = _LSTMNet(n_features, self.HIDDEN_DIM).to(self._device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Build sequences then train for 1 epoch with batch_size=512 and Adam lr=1e-3."""
        Xs, ys = prepare_sequences(X_train, y_train, self.LOOKBACK, self.MAX_TRAIN_SAMPLES)
        self._net.train()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.LR)
        loss_fn = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(
                torch.tensor(Xs, dtype=torch.float32),
                torch.tensor(ys, dtype=torch.float32).unsqueeze(1),
            ),
            batch_size=self.BATCH_SIZE,
            shuffle=True,
        )
        for _ in range(self.EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict; first (lookback-1) rows are filled with NaN (no full window)."""
        Xs, _ = prepare_sequences(X, np.zeros(len(X)), self.LOOKBACK)
        self._net.eval()
        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32).to(self._device)
            preds = self._net(xb).squeeze(1).cpu().numpy()
        out = np.full(len(X), np.nan)
        out[self.LOOKBACK - 1 :] = preds
        return out

    def reset(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self._net = _LSTMNet(self._n_features, self.HIDDEN_DIM).to(self._device)


# ---------------------------------------------------------------------------
# 6. Transformer
# ---------------------------------------------------------------------------

class TransformerModel(BaseModel):
    """Transformer encoder sequence model.

    Input features are linearly projected to ``d_model`` dimensions, augmented
    with sinusoidal positional encodings, passed through a single encoder layer,
    and the last-token representation is mapped to a scalar prediction.

    Parameters
    ----------
    n_features: Number of input features per time step.
    """

    LOOKBACK: int            = 60
    D_MODEL: int             = 32
    NHEAD: int               = 4
    NUM_ENCODER_LAYERS: int  = 1
    DIM_FEEDFORWARD: int     = 64
    DROPOUT: float           = 0.1
    LR: float                = 1e-3
    BATCH_SIZE: int          = 512
    EPOCHS: int              = 1
    MAX_TRAIN_SAMPLES: int   = 12_000
    MAX_VAL_SAMPLES: int     = 3_000

    def __init__(self, n_features: int) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TransformerModel. "
                "Install it with: pip install torch"
            )
        torch.manual_seed(42)
        np.random.seed(42)
        self._n_features = n_features
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = _TransformerNet(
            n_features,
            d_model=self.D_MODEL,
            nhead=self.NHEAD,
            num_encoder_layers=self.NUM_ENCODER_LAYERS,
            dim_feedforward=self.DIM_FEEDFORWARD,
            dropout=self.DROPOUT,
        ).to(self._device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Build sequences then train for 1 epoch with batch_size=512 and Adam lr=1e-3."""
        Xs, ys = prepare_sequences(X_train, y_train, self.LOOKBACK, self.MAX_TRAIN_SAMPLES)
        self._net.train()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.LR)
        loss_fn = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(
                torch.tensor(Xs, dtype=torch.float32),
                torch.tensor(ys, dtype=torch.float32).unsqueeze(1),
            ),
            batch_size=self.BATCH_SIZE,
            shuffle=True,
        )
        for _ in range(self.EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict; first (lookback-1) rows are filled with NaN (no full window)."""
        Xs, _ = prepare_sequences(X, np.zeros(len(X)), self.LOOKBACK)
        self._net.eval()
        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32).to(self._device)
            preds = self._net(xb).squeeze(1).cpu().numpy()
        out = np.full(len(X), np.nan)
        out[self.LOOKBACK - 1 :] = preds
        return out

    def reset(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self._net = _TransformerNet(
            self._n_features,
            d_model=self.D_MODEL,
            nhead=self.NHEAD,
            num_encoder_layers=self.NUM_ENCODER_LAYERS,
            dim_feedforward=self.DIM_FEEDFORWARD,
            dropout=self.DROPOUT,
        ).to(self._device)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(model_name: str) -> BaseModel:
    """Return a fresh model instance for the given name.

    Supported names (case-insensitive):
    - ``"momentum"``  → :class:`MomentumModel`
    - ``"ridge"``     → :class:`RidgeModel`
    - ``"lightgbm"``  → :class:`LightGBMModel`

    Models that require architecture arguments (``graph_mlp``, ``lstm``,
    ``transformer``) must be constructed directly because they need the
    feature-dimension at initialisation time.

    Raises
    ------
    ValueError: Unknown model name or name of a model requiring extra args.
    """
    _SIMPLE = {
        "momentum": MomentumModel,
        "ridge":    RidgeModel,
        "lightgbm": LightGBMModel,
    }
    _NEEDS_ARGS = {"graph_mlp", "lstm", "transformer"}

    name = model_name.lower()
    if name in _SIMPLE:
        return _SIMPLE[name]()
    if name in _NEEDS_ARGS:
        raise ValueError(
            f"'{model_name}' requires feature-dimension arguments. "
            "Construct it directly, e.g. "
            "GraphMLPModel(base_features_dim, relation_features_dim), "
            "LSTMModel(n_features), or TransformerModel(n_features)."
        )
    raise ValueError(
        f"Unknown model name '{model_name}'. "
        f"Available via factory: {sorted(_SIMPLE)}. "
        f"Construct directly: {sorted(_NEEDS_ARGS)}."
    )
