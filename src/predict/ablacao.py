from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from predict import dados, esquema, especificacoes, metricas, modelos, validacao
from predict.ajustar import ler_params
from predict.comparar import metodo_escolhido

VARIANTES = ("A_so_grupos_grandes", "B_todas_as_maquinas", "C_especificacoes", "D_pesos", "E_sem_redundancia")
ESQUEMAS = {"maquina": 1, "familia": 1}
MIN_ANTIGO = 10


def colunas (variante: str, df: pd.DataFrame, alvo: str) -> tuple[str, ...]:
    cols = esquema.colunas(alvo)
    if variante < "C":
        cols = tuple(c for c in cols if c not in especificacoes.COLUNAS)
    if variante >= "E":
        cols = esquema.sem_redundancia(df, cols)
    return cols


def rodar (alvo: str, metodo: str) -> pd.DataFrame:
    df = dados.carregar(alvo)
    dev, _ = validacao.teste_blindado(df)
    d = df.iloc[dev].reset_index(drop=True)
    y = d[esquema.COLUNA_ALVO[alvo]].to_numpy(float)
    grande = ((d["fonte"] != "opendata") | (d["n_medicoes"] >= MIN_ANTIGO)).to_numpy()
    params = ler_params(metodo, alvo)
    g_parada = validacao.grupos_de(d, "maquina")
    linhas = []
    for esquema_cv, sementes in ESQUEMAS.items():
        g = validacao.grupos_de(d, esquema_cv)
        for semente in range(sementes):
            for k, (tr, va) in enumerate(validacao.dobras(g, esquema_cv, 5, semente)):
                va = va[grande[va]]
                for v in VARIANTES:
                    tr_v = tr[grande[tr]] if v < "B" else tr
                    w = d["peso"].to_numpy(float) if v >= "D" else np.ones(len(d))
                    cols = colunas(v, d.iloc[tr_v], alvo)
                    X, cats = esquema.montar_X(d.iloc[tr_v], cols)
                    X_va, _ = esquema.montar_X(d.iloc[va], cols, cats)
                    m = modelos.criar(metodo, alvo, params, semente).ajustar(X, y[tr_v], g_parada[tr_v], w[tr_v])
                    linhas.append({"alvo": alvo, "esquema": esquema_cv, "semente": semente, "dobra": k, "variante": v,
                                   "linhas_treino": len(tr_v), "colunas": len(cols),
                                   **metricas.resumo(y[va], m.prever(X_va))})
            print(f"{alvo} {esquema_cv} semente {semente} pronta", flush=True)
    return pd.DataFrame(linhas)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metodo", default=None)
    ap.add_argument("--alvos", default=",".join(dados.ALVOS))
    args = ap.parse_args()
    metodo = args.metodo or metodo_escolhido()
    r = pd.concat([rodar(a, metodo) for a in args.alvos.split(",")], ignore_index=True)
    destino = dados.PASTA_EXPERIMENTOS / "ablacao.csv"
    r.to_csv(destino, index=False)
    t = r.pivot_table(index=["alvo", "variante"], columns="esquema", values="rmse_log").round(4)
    print(f"rmse_log médio ({metodo}), avaliado nas mesmas linhas:\n{t.to_string()}")
    b = r.pivot_table(index=["alvo", "esquema", "semente", "dobra"], columns="variante", values="rmse_log")
    d = b.diff(axis=1).iloc[:, 1:]
    print("\nmudança pareada para a variante anterior (média; dobras em que melhorou):")
    for (alvo, esq), gr in d.groupby(level=["alvo", "esquema"]):
        print(f"  {alvo:10s} {esq:8s} " + "  ".join(f"{c[:1]}: {gr[c].mean():+.4f} ({int((gr[c] < 0).sum())}/{len(gr)})"
                                                for c in gr.columns))
    print(f"escrito: {destino}")


if __name__ == "__main__":
    main()
