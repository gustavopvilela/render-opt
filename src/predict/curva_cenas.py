"""
Curva de aprendizado em CENAS: o erro numa cena nova cai quando o treino tem mais cenas?

    python -m predict.curva_cenas --repeticoes 5

Para cada cena c e cada k = 2..(n_cenas - 1): sorteia k cenas entre as outras, treina com
elas (máquinas do desenvolvimento) e mede o erro em c (máquinas do desenvolvimento também,
porque a pergunta aqui é a cena, não a máquina). Duas variantes:
  atributos   o esquema da vazão (cena + máquina)
  so_maquina  só as colunas de máquina: a referência de quem não sabe nada da cena
"""

from __future__ import annotations
import argparse
import lightgbm as lgb
import numpy as np
import pandas as pd
from predict import dados, esquema, validacao

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=10,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeticoes", type=int, default=5)
    args = ap.parse_args()

    df = dados.carregar("vazao")
    dev, _ = validacao.teste_blindado(df)
    df = df.iloc[dev].reset_index(drop=True)
    cenas = sorted(df["cena_sha256"].unique())
    nome = df.groupby("cena_sha256")["cena"].first()
    variantes = {"atributos": esquema.colunas("vazao"), "so_maquina": esquema.MAQUINA}
    rng = np.random.default_rng(2026)
    linhas = []
    # Cada cena prevista com k outras cenas no treino
    for c in cenas:
        teste = df[df["cena_sha256"] == c]
        outras = [o for o in cenas if o != c]
        for k in range(2, len(outras) + 1):
            for rep in range(args.repeticoes if k < len(outras) else 1):
                escolha = rng.choice(outras, size=k, replace=False)
                treino = df[df["cena_sha256"].isin(escolha)]
                for v, cols in variantes.items():
                    Xtr, cats = esquema.montar_X(treino, cols)
                    Xte, _ = esquema.montar_X(teste, cols, cats)
                    m = lgb.LGBMRegressor(**PARAMS, random_state=rep).fit(Xtr, treino["log_vazao"], sample_weight=treino["peso"])
                    e = m.predict(Xte) - teste["log_vazao"].to_numpy()
                    linhas.append({"cena": nome[c], "k": k, "rep": rep, "variante": v,
                                   "rmse_log": float(np.sqrt(np.mean(e ** 2))), "mdae_log": float(np.median(np.abs(e)))})
        print(f"{nome[c]} pronta", flush=True)

    r = pd.DataFrame(linhas)
    destino = dados.PASTA_EXPERIMENTOS / "curva_cenas.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(destino, index=False)
    # Resumo: média por cena, depois entre cenas
    t = r.groupby(["variante", "k", "cena"])["mdae_log"].mean().groupby(["variante", "k"]).agg(["mean", "std"])
    print("\nerro mediano em log na cena deixada de fora (média entre cenas ± desvio):")
    print(t.round(3).unstack("variante").to_string())
    print(f"escrito: {destino}")


if __name__ == "__main__":
    main()
