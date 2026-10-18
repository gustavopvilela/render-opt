from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from predict import dados, modelos

SUPERFICIE, TEXTO, TEXTO_2, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CORES = {"lgbm": "#2a78d6", "floresta": "#eb6834", "rede": "#1baf7a"}
ALVO = {"vazao": "vazão", "custo_fixo": "custo fixo"}
ESQUEMA = {"maquina": "máquina nova", "familia": "família nova", "cena": "cena nova"}


def _eixo (ax) -> None:
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRADE)
    ax.tick_params(colors=TEXTO_2, length=0)
    ax.grid(axis="x", color=GRADE, linewidth=1)
    ax.set_axisbelow(True)


def _barras (ax, rotulos: list[str], series: dict[str, np.ndarray], cores: dict[str, str], nomes: dict[str, str]) -> None:
    n = len(series)
    altura = min(0.8 / n, 0.24)
    y = np.arange(len(rotulos))
    for i, (chave, valores) in enumerate(series.items()):
        pos = y - (n - 1) * altura / 2 + i * altura
        ax.barh(pos, valores, height=altura * 0.9, color=cores[chave], label=nomes[chave])
        for p, v in zip(pos, valores):
            if np.isnan(v):
                continue
            ax.text(v, p, f" {v:.3f}", va="center", fontsize=7.5, color=TEXTO_2)
    ax.set_yticks(y, rotulos, color=TEXTO)
    ax.invert_yaxis()


def metodos (pasta: Path) -> None:
    r = pd.read_csv(dados.PASTA_EXPERIMENTOS / "comparacao.csv")
    t = r[(r["recorte"] == "todos") & r["esquema"].isin(ESQUEMA)].groupby(["alvo", "esquema", "metodo"])["rmse_log"].mean()
    rotulos, chaves = [], []
    for a in dados.ALVOS:
        for e in ESQUEMA:
            rotulos.append(f"{ALVO[a]} · {ESQUEMA[e]}")
            chaves.append((a, e))
    series = {m: np.array([t[(a, e, m)] for a, e in chaves]) for m in modelos.METODOS}
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SUPERFICIE)
    _eixo(ax)
    _barras(ax, rotulos, series, CORES, modelos.NOMES)
    ax.set_xlabel("rmse em log (menor é melhor)", color=TEXTO_2)
    ax.set_title("Validação cruzada agrupada: erro médio por cenário", color=TEXTO, loc="left", fontsize=11)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.06)
    ax.legend(frameon=False, loc="upper right", fontsize=8, labelcolor=TEXTO)
    fig.tight_layout()
    fig.savefig(pasta / "metodos_por_cenario.png", dpi=160)
    plt.close(fig)


def ablacao (pasta: Path) -> None:
    r = pd.read_csv(dados.PASTA_EXPERIMENTOS / "ablacao.csv")
    t = r.groupby(["alvo", "esquema", "variante"])["rmse_log"].mean()
    variantes = sorted(r["variante"].unique())
    nomes = {"A_so_grupos_grandes": "A · como antes", "B_todas_as_maquinas": "B · + todas as máquinas",
             "C_especificacoes": "C · + especificações", "D_pesos": "D · + pesos", "E_sem_redundancia": "E · + sem redundância"}
    fig, eixos = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SUPERFICIE, sharey=True)
    for ax, alvo in zip(eixos, dados.ALVOS):
        _eixo(ax)
        for esq, cor in zip([e for e in ESQUEMA if e in set(r["esquema"])], CORES.values()):
            v = [t[(alvo, esq, var)] for var in variantes]
            ax.plot(range(len(variantes)), v, color=cor, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SUPERFICIE, markeredgewidth=1.5, label=ESQUEMA[esq])
            ax.text(len(variantes) - 1, v[-1], f"  {v[-1]:.3f}", va="center", fontsize=7.5, color=TEXTO_2)
        ax.grid(axis="y", color=GRADE, linewidth=1)
        ax.set_xticks(range(len(variantes)), [nomes[v].split(" · ")[0] for v in variantes], color=TEXTO)
        ax.set_title(ALVO[alvo], color=TEXTO, loc="left", fontsize=10)
    eixos[0].set_ylabel("rmse em log", color=TEXTO_2)
    eixos[0].legend(frameon=False, fontsize=8, labelcolor=TEXTO)
    fig.suptitle("Ablação: cada tratamento somado ao anterior (mesmas linhas de avaliação)",
                 color=TEXTO, x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.01, "  ".join(nomes[v] for v in variantes), fontsize=7, color=TEXTO_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(pasta / "ablacao.png", dpi=160)
    plt.close(fig)


def importancia (pasta: Path) -> None:
    r = pd.read_csv(dados.PASTA_EXPERIMENTOS / "importancia.csv")
    g = r[r["nivel"] == "grupo"]
    nomes = sorted(g["nome"].unique())
    t = g.pivot_table(index="nome", columns="alvo", values="piora_rmse_log").reindex(nomes)
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SUPERFICIE)
    _eixo(ax)
    _barras(ax, nomes, {a: t[a].to_numpy() for a in dados.ALVOS},
            {"vazao": CORES["lgbm"], "custo_fixo": CORES["floresta"]}, ALVO)
    ax.set_xlabel("piora do rmse em log ao embaralhar o grupo (teste blindado)", color=TEXTO_2)
    ax.set_title("Importância por permutação, por grupo de atributos", color=TEXTO, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=8, labelcolor=TEXTO)
    fig.tight_layout()
    fig.savefig(pasta / "importancia_grupos.png", dpi=160)
    plt.close(fig)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", type=Path, default=dados.RAIZ / "docs" / "figuras" / "ia-v2")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    for f in (metodos, ablacao, importancia):
        f(args.saida)
    print(f"figuras em {args.saida}")


if __name__ == "__main__":
    main()
