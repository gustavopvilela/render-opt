from __future__ import annotations
import numpy as np


def mediana_ponderada (v: np.ndarray, w: np.ndarray) -> float:
    ordem = np.argsort(v)
    acumulado = np.cumsum(w[ordem]) / w.sum()
    return float(v[ordem][min(int(np.searchsorted(acumulado, 0.5)), len(v) - 1)])


def resumo (y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    r = p - y
    rel = np.abs(np.expm1(r))
    media_y = np.average(y, weights=w)
    ss_tot = float(np.sum(w * (y - media_y) ** 2))
    return {
        "n": int(len(y)),
        "rmse_log": float(np.sqrt(np.average(r ** 2, weights=w))),
        "fator_mediano": float(np.exp(mediana_ponderada(np.abs(r), w))),
        "erro_rel_mediano": mediana_ponderada(rel, w),
        "dentro_20pct": float(np.average(rel <= 0.20, weights=w)),
        "vies_log": mediana_ponderada(r, w),
        "r2_log": float(1 - np.sum(w * r ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }
