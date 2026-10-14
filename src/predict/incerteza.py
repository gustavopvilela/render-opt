"""
Limite pessimista de cada alvo, com calibração conformal.

    python -m predict.incerteza --alvo vazao                 # máquina nova, cena conhecida
    python -m predict.incerteza --alvo vazao --cena-nova     # + cena nova (mais lento)

O escalonador e o vigia de tempo excedido precisam de um limite "quase com
certeza não passa disso". O LightGBM com objective="quantile" dá uma estimativa
do quantil, mas sem garantia. A correção conformal (CQR, Romano et al. 2019)
desloca o quantil com um conjunto de calibração para que a cobertura feche.

Atenção ao sentido: para a VAZÃO, o caso ruim é ela ser BAIXA (o frame demora).
Por isso a vazão usa o quantil 10% (limite inferior); o custo fixo usa o 90%.

Duas folgas (§2.11):
  folga          calibrada em MÁQUINAS que o quantil não viu: vale para cena conhecida;
  folga_cena     calibrada com resíduos de CENAS que o quantil não viu (deixa uma de fora).
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from predict import dados, esquema, modelos, validacao

QUANTIL = {"vazao": (0.10, "inferior"), "custo_fixo": (0.90, "superior")}
NIVEL = 0.90


def ajustar_quantil (X, y, grupos, alvo: str, params: dict | None = None, semente: int = 0, pesos=None):
    alpha, _ = QUANTIL[alvo]
    p = {**(params or {}), "objective": "quantile", "alpha": alpha}
    return modelos.LGBM(p, semente).ajustar(X, y, grupos, pesos)


def escores (q: np.ndarray, y: np.ndarray, alvo: str) -> np.ndarray:
    """Quanto o limite errou para o lado ruim: positivo = não cobriu."""
    _, lado = QUANTIL[alvo]
    return (y - q) if lado == "superior" else (q - y)


def calibrar (q_cal: np.ndarray, y_cal: np.ndarray, alvo: str, nivel: float = NIVEL) -> float:
    """
    Deslocamento conformal unilateral: o quantil ceil((n+1)*nivel)/n dos escores.
    Garante cobertura >= nivel se calibração e uso forem trocáveis (Angelopoulos e Bates, 2021).
    """
    s = escores(q_cal, y_cal, alvo)
    n = len(s)
    k = min(int(np.ceil((n + 1) * nivel)) / n, 1.0)
    return float(np.quantile(s, k, method="higher"))


def _quantil_ponderado (s: np.ndarray, w: np.ndarray, nivel: float) -> float:
    ordem = np.argsort(s)
    acumulado = np.cumsum(w[ordem]) / w.sum()
    i = min(int(np.searchsorted(acumulado, nivel)), len(s) - 1)
    return float(s[ordem][i])


def folga_cena_nova (X, y, g_maquina, g_cena, alvo: str, params: dict | None = None,
                     pesos=None, nivel: float = NIVEL) -> float:
    """
    Folga para cena nova: cada cena é prevista por um quantil treinado SEM ela, e os
    escores de todas as cenas formam a calibração. Cada cena pesa igual (1/n_linhas),
    para as cenas com milhares de máquinas não decidirem sozinhas. Com poucas cenas
    isto é uma aproximação inspirada no CV+ (Barber et al., 2021), não a garantia dele.
    """
    s_todos, w_todos = [], []
    for tr, va in validacao.dobras(g_cena, "cena"):
        m = ajustar_quantil(X.iloc[tr], y[tr], g_maquina[tr], alvo, params,
                            pesos=None if pesos is None else pesos[tr])
        s_todos.append(escores(m.prever(X.iloc[va]), y[va], alvo))
        w_todos.append(np.full(len(va), 1.0 / len(va)))
    return _quantil_ponderado(np.concatenate(s_todos), np.concatenate(w_todos), nivel)


def aplicar (q: np.ndarray, deslocamento: float, alvo: str) -> np.ndarray:
    _, lado = QUANTIL[alvo]
    return q + deslocamento if lado == "superior" else q - deslocamento


def cobre (y: np.ndarray, limite: np.ndarray, alvo: str) -> np.ndarray:
    _, lado = QUANTIL[alvo]
    return y <= limite if lado == "superior" else y >= limite


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alvo", choices=tuple(QUANTIL), default="vazao")
    ap.add_argument("--cena-nova", action="store_true", help="avalia também a folga de cena nova")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    df = dados.carregar(args.alvo, args.dataset)
    dev, _ = validacao.teste_blindado(df)
    df = df.iloc[dev].reset_index(drop=True)
    gm, gc = validacao.grupos_de(df, "maquina"), validacao.grupos_de(df, "cena")
    X, _ = esquema.montar_X(df, esquema.colunas(args.alvo))
    y = df[esquema.COLUNA_ALVO[args.alvo]].to_numpy(float)
    w = df["peso"].to_numpy(float)

    # Máquina nova, cena conhecida
    linhas = []
    for k, (tr, va) in enumerate(validacao.dobras(gm, "maquina", 5, semente=0)):
        sub_tr, sub_cal = validacao.separar_teste(gm[tr], 0.2, semente=k)
        tr_q, cal = tr[sub_tr], tr[sub_cal]
        m = ajustar_quantil(X.iloc[tr_q], y[tr_q], gm[tr_q], args.alvo, pesos=w[tr_q])
        desloc = calibrar(m.prever(X.iloc[cal]), y[cal], args.alvo)
        q_va = m.prever(X.iloc[va])
        linhas.append(pd.DataFrame({
            "tipo": df["dispositivo_tipo"].iloc[va].to_numpy(),
            "cru": cobre(y[va], q_va, args.alvo),
            "conformal": cobre(y[va], aplicar(q_va, desloc, args.alvo), args.alvo),
            "folga_log": np.full(len(va), desloc),
        }))
    r = pd.concat(linhas)
    tabela = r.groupby("tipo")[["cru", "conformal"]].mean()
    tabela.loc["todos"] = r[["cru", "conformal"]].mean()
    tabela["n"] = r.groupby("tipo").size().reindex(tabela.index).fillna(len(r)).astype(int)
    print(f"[máquina nova] cobertura desejada {NIVEL:.0%}  (quantil {QUANTIL[args.alvo][0]:.2f}, lado {QUANTIL[args.alvo][1]})")
    print(tabela.round(3).to_string())
    print(f"folga conformal média (log): {r['folga_log'].mean():.4f}")

    if not args.cena_nova:
        return
    # Cena nova
    print(f"\n[cena nova] cobertura por cena deixada de fora (desejada {NIVEL:.0%})")
    for tr, va in validacao.dobras(gc, "cena"):
        folga_c = folga_cena_nova(X.iloc[tr], y[tr], gm[tr], gc[tr], args.alvo, pesos=w[tr])
        m = ajustar_quantil(X.iloc[tr], y[tr], gm[tr], args.alvo, pesos=w[tr])
        q_va = m.prever(X.iloc[va])
        print(f"  {df['cena'].iloc[va[0]]:14s} n={len(va):5d}  cru {cobre(y[va], q_va, args.alvo).mean():.3f}"
              f"  com folga de cena nova ({folga_c:.3f}) {cobre(y[va], aplicar(q_va, folga_c, args.alvo), args.alvo).mean():.3f}")


if __name__ == "__main__":
    main()
