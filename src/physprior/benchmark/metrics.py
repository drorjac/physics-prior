"""Metrics. Six axes, because "accuracy" alone hides the whole story."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if not np.all(np.isfinite(y_pred)):
        return float("inf")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, scale: float | None = None) -> float:
    """RMSE in units of the spread of the truth, so tracks are comparable."""
    s = float(np.std(np.asarray(y_true, float))) if scale is None else scale
    if s <= 0:
        return float("nan")
    return rmse(y_true, y_pred) / s


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = y_true != 0
    if not np.all(np.isfinite(y_pred)):
        return float("inf")
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100.0)


def chi2_reduced(
    y: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray, n_params: int
) -> float:
    r = (np.asarray(y, float) - np.asarray(y_pred, float)) / np.asarray(sigma, float)
    dof = max(len(r) - n_params, 1)
    return float(np.sum(r**2) / dof)


def param_report(
    name: str, hat: float, published: float, sigma: float | None = None
) -> dict:
    """How close is a recovered physical constant to the published value?"""
    out = {
        "parameter": name,
        "recovered": float(hat),
        "published": float(published),
        "rel_error_ppm": float((hat - published) / published * 1e6),
        "rel_error_pct": float((hat - published) / published * 100.0),
    }
    out["sigma"] = float(sigma) if sigma is not None and np.isfinite(sigma) else None
    out["n_sigma"] = (
        float((hat - published) / sigma)
        if sigma is not None and np.isfinite(sigma) and sigma > 0
        else None
    )
    return out
