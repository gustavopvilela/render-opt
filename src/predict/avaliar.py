from __future__ import annotations
import argparse
import time
import pandas as pd
from predict import dados, metricas, modelos, validacao
from predict.ajustar import ler_params

CSV = dados.PASTA_EXPERIMENTOS / "comparacao.csv"
ESQUEMAS = ("maquina", "familia", "cena", "aleatorio")
SEMENTES = {"maquina": 3, "familia": 3, "cena": 1, "aleatorio": 1}


def recortes (df: pd.DataFrame, esquema_cv: str) -> pd.Series:
    if esquema_cv == "cena":
        return "cena:" + validacao.texto(df["cena"])
    return "tipo:" + validacao.texto(df["dispositivo_tipo"])


def avaliar (alvo: str, esquemas: list[str], metodos: list[str]) -> pd.DataFrame:
    dev, _ = dados.separar(alvo)
    g_parada = dev.grupos("maquina")
    params = {m: ler_params(m, alvo) for m in metodos}
    linhas = []
    for esquema_cv in esquemas:
        g = dev.grupos(esquema_cv)
        rot = recortes(dev.df, esquema_cv).to_numpy()
        for semente in range(SEMENTES[esquema_cv]):
            for k, (tr, va) in enumerate(validacao.dobras(g, esquema_cv, 5, semente)):
                for metodo in metodos:
                    inicio = time.perf_counter()
                    m = modelos.criar(metodo, alvo, params[metodo], semente)
                    m.ajustar(dev.X.iloc[tr], dev.y[tr], g_parada[tr], dev.w[tr])
                    p = m.prever(dev.X.iloc[va])
                    base = {"alvo": alvo, "esquema": esquema_cv, "semente": semente, "dobra": k, "metodo": metodo,
                            "segundos": time.perf_counter() - inicio}
                    linhas.append({**base, "recorte": "todos", **metricas.resumo(dev.y[va], p, dev.w[va])})
                    for r in sorted(set(rot[va])):
                        s = rot[va] == r
                        linhas.append({**base, "recorte": r, **metricas.resumo(dev.y[va][s], p[s], dev.w[va][s])})
            print(f"{alvo} {esquema_cv} semente {semente} pronta", flush=True)
    return pd.DataFrame(linhas)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alvos", default=",".join(dados.ALVOS))
    ap.add_argument("--esquemas", default=",".join(ESQUEMAS))
    ap.add_argument("--metodos", default=",".join(("B2",) + modelos.METODOS))
    ap.add_argument("--saida", default=str(CSV))
    args = ap.parse_args()

    r = pd.concat([avaliar(a, args.esquemas.split(","), args.metodos.split(",")) for a in args.alvos.split(",")],
                  ignore_index=True)
    dados.PASTA_EXPERIMENTOS.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.saida, index=False)
    todos = r[r["recorte"] == "todos"]
    print(todos.pivot_table(index=["alvo", "metodo"], columns="esquema", values="rmse_log").round(4).to_string())
    print(f"escrito: {args.saida}")


if __name__ == "__main__":
    main()
