from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from predict import dados, esquema, metricas, modelos, piso, validacao

TAU2_MINIMO = 1e-3


def fator_ruido (n: np.ndarray) -> np.ndarray:
    return 1.0 / np.asarray(n, float)


def tau2_piloto (df: pd.DataFrame, alvo: str, sigma2: float) -> tuple[float, float]:
    dev, _ = validacao.teste_blindado(df)
    d = df.iloc[dev].reset_index(drop=True)
    cols = esquema.sem_redundancia(d, esquema.colunas(alvo))
    X, _ = esquema.montar_X(d, cols)
    y = d[esquema.COLUNA_ALVO[alvo]].to_numpy(float)
    g = validacao.grupos_de(d, "maquina")
    r2, ruido = [], []
    for tr, va in validacao.dobras(g, "maquina", 5, semente=0):
        m = modelos.criar("lgbm", alvo).ajustar(X.iloc[tr], y[tr], g[tr])
        r2.append((m.prever(X.iloc[va]) - y[va]) ** 2)
        ruido.append(sigma2 * fator_ruido(d["n_obs"].to_numpy()[va]))
    r2, ruido = np.concatenate(r2), np.concatenate(ruido)
    return max(float(r2.mean() - ruido.mean()), TAU2_MINIMO), float(np.sqrt(r2.mean()))


def pesos (n_obs: np.ndarray, peso_base: np.ndarray, sigma2: float, tau2: float) -> np.ndarray:
    w = 1.0 / (tau2 + sigma2 * fator_ruido(n_obs))
    w = w / w.mean()
    return w * np.asarray(peso_base, float)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    resumo = {}
    for alvo, alvo_piso in (("vazao", "vazao"), ("custo_fixo", "custo_fixo")):
        df = pd.read_parquet(dados.DATASET[alvo])
        df = df[df[esquema.COLUNA_ALVO[alvo]].notna()].reset_index(drop=True)
        sigma = piso.calcular(piso.CSV_SEM_AGREGAR, alvo_piso, 3)["sigma_ruido_log"]
        tau2, rmse = tau2_piloto(df, alvo, sigma ** 2)
        df["peso"] = pesos(df["n_obs"], df["peso_base"], sigma ** 2, tau2)
        df.to_parquet(dados.DATASET[alvo], index=False)
        k = sigma ** 2 / tau2
        por_n = df.groupby(pd.cut(df["n_obs"], [0, 1, 2, 9, np.inf], labels=["1", "2", "3-9", ">=10"]),
                           observed=True)["peso"].mean()
        resumo[alvo] = {"sigma_ruido_log": sigma, "tau2": tau2, "rmse_piloto": rmse, "k": k,
                        "peso_medio_por_n": {str(i): float(v) for i, v in por_n.items()}}
        print(f"{alvo:10s} sigma {sigma:.3f}  tau² {tau2:.4f} (piloto rmse {rmse:.3f})  k = {k:.3f}  "
              f"peso médio por n: {por_n.round(3).to_dict()}")
    destino = dados.PASTA_EXPERIMENTOS / "pesos.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    print(f"escrito: {destino}")


if __name__ == "__main__":
    main()
