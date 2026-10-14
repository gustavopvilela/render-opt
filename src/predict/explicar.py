"""
Figuras de importância de atributos (SHAP) e de resíduos, para a monografia (RF-23).

    python -m predict.explicar --alvo vazao

O SHAP é calculado no TESTE BLINDADO: no treino ele explicaria a memorização.
Ele explica o MODELO, não a física do render; com atributos correlacionados,
a importância se divide entre eles. Com 8 cenas, a importância dos atributos
de CENA não deve ser interpretada (§2.4).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from predict import dados, esquema, modelos, validacao


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alvo", choices=("vazao", "custo_fixo"), default="vazao")
    ap.add_argument("--params", type=Path, default=None)
    args = ap.parse_args()

    df = dados.carregar(args.alvo)
    cols = esquema.colunas(args.alvo)
    y = df[esquema.COLUNA_ALVO[args.alvo]].to_numpy(float)
    w = df["peso"].to_numpy(float)
    grupos = validacao.grupos_de(df, "maquina")
    dev, teste = validacao.teste_blindado(df)
    X_dev, cats = esquema.montar_X(df.iloc[dev], cols)
    X_te, _ = esquema.montar_X(df.iloc[teste], cols, cats)
    params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else None
    # Modelo treinado fora do teste blindado
    m = modelos.criar("lgbm", args.alvo, params).ajustar(X_dev, y[dev], grupos[dev], w[dev])

    pasta = dados.PASTA_EXPERIMENTOS / "figuras"
    pasta.mkdir(parents=True, exist_ok=True)

    # SHAP no teste blindado
    valores = shap.TreeExplainer(m.modelo.booster_).shap_values(X_te)
    plt.figure()
    shap.summary_plot(valores, X_te, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(pasta / f"shap_{args.alvo}.png", dpi=150)
    plt.close("all")

    importancia = np.abs(valores).mean(axis=0)
    ordem = np.argsort(-importancia)[:15]
    cena = set(esquema.CENA_VAZAO + esquema.CENA_CUSTO)
    print("importância média |SHAP| (log do alvo); [cena] = atributo de cena:")
    for i in ordem:
        print(f"  {cols[i]:28s} {importancia[i]:.4f} {'[cena]' if cols[i] in cena else ''}")

    # Resíduos no teste blindado
    p = m.prever(X_te)
    tipos = df["dispositivo_tipo"].iloc[teste].astype(str).to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for t in sorted(set(tipos)):
        sel = tipos == t
        ax[0].scatter(y[teste][sel], p[sel], s=8, alpha=0.6, label=t)
    lim = [min(y[teste].min(), p.min()), max(y[teste].max(), p.max())]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("real (log)")
    ax[0].set_ylabel("previsto (log)")
    ax[0].legend(fontsize=8)
    ax[1].hist(p - y[teste], bins=50)
    ax[1].set_xlabel("resíduo (previsto - real, log)")
    fig.tight_layout()
    fig.savefig(pasta / f"residuos_{args.alvo}.png", dpi=150)
    print(f"figuras em {pasta}")


if __name__ == "__main__":
    main()
