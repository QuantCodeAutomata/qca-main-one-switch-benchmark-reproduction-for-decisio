"""Tests for src/models.py – model interfaces and hyperparameters."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import LightGBMModel, MomentumModel, RidgeModel, get_model


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_train_test(
    n_train: int = 200,
    n_test: int  = 50,
    n_feat: int  = 10,
    seed: int    = 42,
):
    rng     = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    y_train = (X_train @ rng.standard_normal(n_feat)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  n_feat)).astype(np.float32)
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# MomentumModel
# ---------------------------------------------------------------------------

def test_momentum_uses_ret20():
    """MomentumModel.predict returns the first column of the feature matrix.

    In the feature order produced by compute_base_features the momentum signal
    is placed at column index 0 as documented in the model's docstring.
    Concretely the runner passes features such that the desired momentum column
    is at position 0; this test verifies the slicing contract.
    """
    model    = MomentumModel()
    n_samples, n_feat = 8, 10

    X = np.arange(n_samples * n_feat, dtype=float).reshape(n_samples, n_feat)
    # Mark col 0 with a unique pattern
    X[:, 0] = np.arange(n_samples) * 100.0

    preds = model.predict(X)

    np.testing.assert_array_equal(
        preds, X[:, 0],
        err_msg="MomentumModel should return the first feature column (index 0)",
    )


def test_momentum_fit_is_noop():
    """Calling fit on MomentumModel must not raise and must not change predict output."""
    model = MomentumModel()
    rng   = np.random.default_rng(1)
    X     = rng.standard_normal((20, 5)).astype(np.float32)
    y     = rng.standard_normal(20).astype(np.float32)

    preds_before = model.predict(X).copy()
    model.fit(X, y)
    preds_after  = model.predict(X)

    np.testing.assert_array_equal(preds_before, preds_after)


# ---------------------------------------------------------------------------
# RidgeModel
# ---------------------------------------------------------------------------

def test_ridge_fit_predict_shape():
    """Ridge fit/predict round-trip: output shape = (n_test,)."""
    X_train, y_train, X_test = _make_train_test()
    model = RidgeModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (50,), f"Expected (50,), got {preds.shape}"
    assert not np.any(np.isnan(preds)), "Ridge predictions contain NaN"
    assert not np.any(np.isinf(preds)), "Ridge predictions contain Inf"


def test_ridge_alpha():
    """RidgeModel must use alpha = 1.0 as specified in the benchmark."""
    model = RidgeModel()
    assert model._model.alpha == 1.0, (
        f"Ridge alpha should be 1.0, got {model._model.alpha}"
    )


def test_ridge_reset_clears_weights():
    """After reset(), a fresh fit on different data changes predictions."""
    X_train, y_train, X_test = _make_train_test(seed=0)
    X_train2, y_train2, _    = _make_train_test(seed=99)

    model = RidgeModel()
    model.fit(X_train, y_train)
    preds1 = model.predict(X_test).copy()

    model.reset()
    model.fit(X_train2, y_train2)
    preds2 = model.predict(X_test)

    assert not np.allclose(preds1, preds2), (
        "After reset and different training, predictions should change"
    )


# ---------------------------------------------------------------------------
# LightGBMModel
# ---------------------------------------------------------------------------

def test_lgbm_fit_predict_shape():
    """LightGBM fit/predict round-trip: output shape = (n_test,)."""
    X_train, y_train, X_test = _make_train_test()
    model = LightGBMModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (50,), f"Expected (50,), got {preds.shape}"
    assert not np.any(np.isnan(preds)), "LightGBM predictions contain NaN"


def test_lgbm_predictions_finite():
    """LightGBM should produce finite predictions on an in-distribution test set."""
    X_train, y_train, X_test = _make_train_test(n_train=150, n_test=30, seed=7)
    model = LightGBMModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert np.all(np.isfinite(preds)), "LightGBM produced non-finite predictions"


# ---------------------------------------------------------------------------
# get_model factory
# ---------------------------------------------------------------------------

def test_model_factory_returns_correct_type():
    """get_model('X') returns an instance of the expected class."""
    assert isinstance(get_model("Momentum"),  MomentumModel)
    assert isinstance(get_model("Ridge"),     RidgeModel)
    assert isinstance(get_model("LightGBM"),  LightGBMModel)


def test_model_factory_case_insensitive():
    """Factory lookup is case-insensitive per the docstring."""
    assert isinstance(get_model("momentum"),  MomentumModel)
    assert isinstance(get_model("ridge"),     RidgeModel)
    assert isinstance(get_model("lightgbm"),  LightGBMModel)


def test_model_factory_unknown_raises():
    """Requesting an unknown model name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("NonExistentModel")


def test_model_factory_dim_required_raises():
    """Requesting a dim-requiring model via factory should raise ValueError."""
    with pytest.raises(ValueError):
        get_model("lstm")
    with pytest.raises(ValueError):
        get_model("transformer")
    with pytest.raises(ValueError):
        get_model("graph_mlp")
