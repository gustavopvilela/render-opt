"""
O OpenBenchmarking ajuda a prever máquinas novas do Open Data?

    python -m predict.comparar_fontes

Mesmas dobras (agrupadas por máquina, nome normalizado) para todas as variantes; o erro é
medido SÓ nas linhas do Open Data da dobra de teste. O que muda é o que entra no treino:
  so_opendata      as linhas do Open Data
  com_ob           + as do OB, com a vazão derivada do custo fixo imputado
  com_ob_c0        + as do OB, com o limite inferior da vazão (custo fixo = 0)
A comparação é pareada: mesma dobra, mesma semente.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from predict.dataset import SAIDA_VAZAO
from predict.esquema import colunas, montar_X

PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=10,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def alvo_treino (d: pd.DataFrame, variante: str) -> np.ndarray:
    if variante == "com_ob_c0":
        return np.where(d["fonte"] == "openbenchmarking", d["log_vazao_limite_inferior"], d["log_vazao"])
    return d["log_vazao"].to_numpy()


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=SAIDA_VAZAO)
    ap.add_argument("--sementes", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    df = df[df["fonte"].isin(["opendata", "openbenchmarking"])].reset_index(drop=True)
    cols = colunas("vazao")
    variantes = ("so_opendata", "com_ob", "com_ob_c0")
    rmse = {v: [] for v in variantes}

    # Mesmas dobras por máquina para as três variantes
    for semente in range(args.sementes):
        gkf = GroupKFold(n_splits=5, shuffle=True, random_state=semente)
        for tr, te in gkf.split(df, groups=df["maquina"]):
            teste = df.iloc[te]
            teste = teste[teste["fonte"] == "opendata"]
            for v in variantes:
                treino = df.iloc[tr]
                if v == "so_opendata":
                    treino = treino[treino["fonte"] == "opendata"]
                Xtr, cats = montar_X(treino, cols)
                Xte, _ = montar_X(teste, cols, cats)
                m = lgb.LGBMRegressor(**PARAMS, random_state=semente).fit(Xtr, alvo_treino(treino, v))
                e = m.predict(Xte) - teste["log_vazao"].to_numpy()
                rmse[v].append(float(np.sqrt(np.mean(e ** 2))))

    # Comparação pareada
    base = np.array(rmse["so_opendata"])
    print(f"{'variante':12s}  rmse (log)        Δ pareado vs so_opendata   dobras melhores")
    for v in variantes:
        a = np.array(rmse[v])
        d = a - base
        extra = "" if v == "so_opendata" else f"{d.mean():+.4f} ± {d.std(ddof=1):.4f}        {int((d < 0).sum())}/{len(d)}"
        print(f"{v:12s}  {a.mean():.4f} ± {a.std(ddof=1):.4f}   {extra}")


if __name__ == "__main__":
    main()
