from __future__ import annotations
import argparse
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from predict import dados, modelos
from predict.avaliar import CSV

CENARIOS = ("maquina", "familia", "cena")
DESTINO = dados.PASTA_EXPERIMENTOS / "escolha.json"


def holm (p: dict) -> dict:
    ordem = sorted(p, key=p.get)
    ajustado, maior = {}, 0.0
    for i, chave in enumerate(ordem):
        maior = max(maior, min(1.0, (len(ordem) - i) * p[chave]))
        ajustado[chave] = maior
    return ajustado


def blocos (r: pd.DataFrame) -> pd.DataFrame:
    t = r[(r["recorte"] == "todos") & r["esquema"].isin(CENARIOS) & r["metodo"].isin(modelos.METODOS)]
    return t.pivot_table(index=["alvo", "esquema", "semente", "dobra"], columns="metodo", values="rmse_log")[list(modelos.METODOS)]


def escolher (r: pd.DataFrame) -> dict:
    b = blocos(r)
    postos = b.rank(axis=1)
    por_cenario = postos.groupby(level=["alvo", "esquema"]).mean()
    geral = por_cenario.mean()
    rmse_maquina = b.xs("maquina", level="esquema").mean()
    ordem = sorted(modelos.METODOS, key=lambda m: (geral[m], rmse_maquina[m]))

    s0 = b.xs(0, level="semente")
    friedman = friedmanchisquare(*[s0[m].to_numpy() for m in modelos.METODOS])
    p = {f"{a}-{c}": float(wilcoxon(s0[a], s0[c]).pvalue) for a, c in itertools.combinations(modelos.METODOS, 2)}
    return {
        "metodo": ordem[0],
        "ordem": ordem,
        "posto_geral": geral.round(4).to_dict(),
        "posto_por_cenario": {f"{a}/{e}": v.round(4).to_dict() for (a, e), v in por_cenario.iterrows()},
        "rmse_medio_por_cenario": {f"{a}/{e}": v.round(4).to_dict()
                                   for (a, e), v in b.groupby(level=["alvo", "esquema"]).mean().iterrows()},
        "friedman": {"estatistica": float(friedman.statistic), "p": float(friedman.pvalue), "blocos": int(len(s0))},
        "wilcoxon_holm": {k: {"p": p[k], "p_holm": v} for k, v in holm(p).items()},
    }


def tabela (r: pd.DataFrame) -> pd.DataFrame:
    t = r[r["recorte"] == "todos"]
    g = t.groupby(["alvo", "esquema", "metodo"], sort=False)
    return pd.DataFrame({
        "rmse_log": g["rmse_log"].mean(), "dp": g["rmse_log"].std(),
        "fator_mediano": g["fator_mediano"].mean(), "dentro_20pct": g["dentro_20pct"].mean(),
        "vies_log": g["vies_log"].mean(), "segundos": g["segundos"].mean(), "dobras": g.size(),
    })


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=str(CSV))
    args = ap.parse_args()
    r = pd.read_csv(args.entrada)

    print(tabela(r).round(4).to_string())
    e = escolher(r)
    print("\nposto médio por cenário (1 = melhor):")
    print(pd.DataFrame(e["posto_por_cenario"]).T.to_string())
    print(f"\nposto geral: {e['posto_geral']}")
    print(f"Friedman ({e['friedman']['blocos']} blocos): chi² = {e['friedman']['estatistica']:.2f}, p = {e['friedman']['p']:.3g}")
    for par, v in e["wilcoxon_holm"].items():
        print(f"  Wilcoxon {par:16s} p = {v['p']:.3g}   Holm: {v['p_holm']:.3g}")
    print(f"\nescolhido: {e['metodo']} ({modelos.NOMES[e['metodo']]})")
    DESTINO.write_text(json.dumps(e, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito: {DESTINO}")


def metodo_escolhido () -> str:
    if not DESTINO.exists():
        raise SystemExit(f"Rode antes o predict.comparar (falta {DESTINO})")
    return json.loads(DESTINO.read_text(encoding="utf-8"))["metodo"]


if __name__ == "__main__":
    main()
