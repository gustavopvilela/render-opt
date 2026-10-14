"""
Métricas de predição. Os modelos trabalham em log; as métricas saem nas duas escalas.

Convenção: y e p são log do alvo (real e previsto). Erro relativo = e^(p - y) - 1.
"""

from __future__ import annotations
import numpy as np


def resumo (y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    r = p - y
    rel = np.abs(np.expm1(r))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "rmse_log": float(np.sqrt(np.mean(r ** 2))),
        "fator_mediano": float(np.exp(np.median(np.abs(r)))),
        "erro_rel_mediano": float(np.median(rel)),
        "mape": float(np.mean(rel)),
        "dentro_20pct": float(np.mean(rel <= 0.20)),
        "vies_log": float(np.median(r)),
        "r2_log": float(1 - np.sum(r ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def desfazer_log (p: np.ndarray, residuos_treino: np.ndarray | None = None) -> np.ndarray:
    """
    Volta para a escala original.

    Sem resíduos: e^p, que estima a MEDIANA do alvo (é o que o escalonador quer).
    Com os resíduos do treino: estimador de smearing de Duan (1983), que estima a MÉDIA.
    """
    base = np.exp(p)
    if residuos_treino is None:
        return base
    return base * float(np.mean(np.exp(residuos_treino)))
