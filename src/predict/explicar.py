from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from predict import dados, esquema, especificacoes, metricas, modelos
from predict.ajustar import ler_params
from predict.comparar import metodo_escolhido

REPETICOES = 5


def grupo_de (coluna: str) -> str:
    if coluna in esquema.CENA_VAZAO + esquema.CENA_CUSTO:
        return "cena"
    if coluna in especificacoes.COLUNAS:
        return "máquina: especificações"
    if coluna.startswith("versao_blender"):
        return "versão do Blender"
    if coluna == "etapa":
        return "etapa"
    return "máquina: nome e sistema"


def piora (m, X: pd.DataFrame, y, w, cols: list[str], rng) -> float:
    base = metricas.resumo(y, m.prever(X), w)["rmse_log"]
    deltas = []
    for _ in range(REPETICOES):
        Xp = X.copy()
        ordem = rng.permutation(len(X))
        for c in cols:
            Xp[c] = X[c].to_numpy()[ordem]
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                Xp[c] = pd.Categorical(Xp[c], categories=X[c].cat.categories)
        deltas.append(metricas.resumo(y, m.prever(Xp), w)["rmse_log"] - base)
    return float(np.mean(deltas))


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metodo", default=None)
    args = ap.parse_args()
    metodo = args.metodo or metodo_escolhido()
    rng = np.random.default_rng(0)
    linhas = []
    for alvo in dados.ALVOS:
        dev, teste = dados.separar(alvo)
        m = modelos.criar(metodo, alvo, ler_params(metodo, alvo)).ajustar(dev.X, dev.y, dev.grupos("maquina"), dev.w)
        grupos = pd.Series({c: grupo_de(c) for c in dev.cols})
        for g, cols in grupos.groupby(grupos).groups.items():
            linhas.append({"alvo": alvo, "nivel": "grupo", "nome": g, "colunas": len(cols),
                           "piora_rmse_log": piora(m, teste.X, teste.y, teste.w, list(cols), rng)})
        for c in dev.cols:
            linhas.append({"alvo": alvo, "nivel": "coluna", "nome": c, "colunas": 1, "grupo": grupos[c],
                           "piora_rmse_log": piora(m, teste.X, teste.y, teste.w, [c], rng)})
        print(f"{alvo} pronto", flush=True)
    r = pd.DataFrame(linhas)
    destino = dados.PASTA_EXPERIMENTOS / "importancia.csv"
    r.to_csv(destino, index=False)
    for alvo, t in r.groupby("alvo"):
        print(f"\n{alvo} ({metodo}): piora do rmse_log ao embaralhar")
        print(t[t["nivel"] == "grupo"].sort_values("piora_rmse_log", ascending=False)[["nome", "colunas", "piora_rmse_log"]]
              .round(4).to_string(index=False))
        print(t[t["nivel"] == "coluna"].nlargest(12, "piora_rmse_log")[["nome", "grupo", "piora_rmse_log"]]
              .round(4).to_string(index=False))
    print(f"escrito: {destino}")


if __name__ == "__main__":
    main()
