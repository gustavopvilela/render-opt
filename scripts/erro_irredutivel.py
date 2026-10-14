#!/usr/bin/env python3
"""Mede o erro irredutivel do problema e gera as figuras de diagnostico.

Motivacao
---------
Um MAPE de 12% nao significa nada sozinho. Ele so vira resultado quando
comparado com o piso: quanto varia o tempo de render da MESMA cena, na MESMA
maquina, entre submissoes diferentes? Essa variacao nao vem da cena nem do
hardware -- vem de estado termico, driver, carga de fundo, versao do Blender.
Nenhum modelo pode explica-la, porque as features sao identicas nas duas
medicoes. Ela e o limite inferior do erro de qualquer preditor.

Este script estima esse piso a partir do dataset NAO agregado (uma linha por
submissao) e escreve as figuras em docs/figuras/.

Metodo
------
Para cada grupo -- (maquina, cena) e, por padrao, tambem a versao maior do
Blender -- com pelo menos --min-medicoes submissoes:

  1. o "preditor oraculo" e a mediana do grupo: e o melhor palpite possivel
     para um modelo que acertasse features e hardware perfeitamente;
  2. o erro de cada submissao e |x - mediana| / mediana;
  3. o piso e a mediana desses erros medios entre grupos.

Tres decisoes de robustez, todas necessarias e todas verificadas nos dados:

* **Agrupar por versao do Blender.** OptiX ficou muito mais rapido entre o
  Blender 3 e o 5. Comparar submissoes da mesma placa em versoes diferentes
  mede evolucao de software, nao ruido -- e a versao e uma feature disponivel
  ao modelo, entao nao pertence ao piso. Desligavel com --sem-versao.

* **Descartar sentinelas.** Existem submissoes com number_of_samples = 2^24
  (16.777.216) concluidas em 3,7 s, o que e fisicamente impossivel: e um
  contador corrompido do launcher. Sao 6 em 1,02 milhao, mas cada uma sozinha
  desloca a media do seu grupo em ordens de grandeza.

* **Descartar renders abortados.** Alguns grupos -- concentrados em iGPU AMD e
  GPU Apple no Blender 3 -- tem amplitude max/min de 10^4 a 10^6 na mesma cena
  e no mesmo hardware. Isso nao e termia: e render que quebrou e reportou
  mesmo assim. O corte em --corte-outlier vezes a mediana os remove; a
  contagem descartada e sempre reportada, para o numero ser auditavel.

E o piso e reportado como **mediana** entre grupos, nao media. A distribuicao
de ruido tem cauda direita pesada, e a media dela descreve os piores grupos,
nao o grupo tipico. A media aparece no relatorio ao lado, justamente para
tornar a cauda visivel.

Uso
---
    python scripts/erro_irredutivel.py
    python scripts/erro_irredutivel.py --min-medicoes 30 --sem-figuras

Ver docs/alvo-do-modelo.md (secoes 5 e 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # sem display: roda em terminal e em CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# O console do Windows costuma ser cp1252 e engasga com acentos.
for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

RAIZ = Path(__file__).resolve().parent.parent
CSV_PADRAO = RAIZ / "data" / "processed" / "dataset_opendata_sem_agregar.csv"
DIR_FIGURAS = RAIZ / "docs" / "figuras"

# So estas colunas sao lidas do CSV de 110 colunas e 1 milhao de linhas.
COLUNAS = ["maquina", "cena", "dispositivo_tipo", "versao_blender_major",
           "medido_num_amostras", "vazao_amostras_pixel_s", "custo_fixo_s"]

ALVOS = {
    "vazao_amostras_pixel_s": "vazão",
    "custo_fixo_s": "custo fixo",
}

# number_of_samples corrompido do launcher (2^24). Ver docstring.
SENTINELA_AMOSTRAS = 2 ** 24

# Paleta fixa por tipo de dispositivo, para que todas as figuras concordem.
CORES = {
    "CPU": "#4C72B0", "OPTIX": "#55A868", "CUDA": "#C44E52",
    "HIP": "#DD8452", "METAL": "#8172B3", "ONEAPI": "#937860",
}


# ---------------------------------------------------------------------------
# Calculo
# ---------------------------------------------------------------------------

def carregar(caminho: Path) -> pd.DataFrame:
    print(f"lendo {caminho.name} ...")
    df = pd.read_csv(caminho, usecols=COLUNAS)
    n = len(df)
    sentinela = df["medido_num_amostras"] >= SENTINELA_AMOSTRAS
    df = df[~sentinela]
    print(f"  {n:,d} submissões, {df['maquina'].nunique():,d} máquinas"
          .replace(",", "."))
    if sentinela.any():
        print(f"  {sentinela.sum()} descartadas: number_of_samples = 2^24 "
              f"(contador corrompido)")
    return df


def preparar(df: pd.DataFrame, alvo: str, chaves: list[str],
             min_medicoes: int, corte: float) -> tuple[pd.DataFrame, int]:
    """Anota cada submissão com a mediana do seu grupo e remove abortadas.

    Devolve (submissões válidas, quantas foram descartadas pelo corte).
    """
    d = df[df[alvo].notna() & (df[alvo] > 0)].copy()

    # Primeira passada: mediana provisória, usada só para localizar os renders
    # quebrados. A mediana aguenta isso mesmo com os outliers ainda presentes
    # -- é exatamente por isso que ela é a estatística escolhida aqui.
    med = d.groupby(chaves, observed=True)[alvo].transform("median")
    razao = d[alvo] / med
    valida = (razao <= corte) & (razao >= 1.0 / corte)
    descartadas = int((~valida).sum())
    d = d[valida]

    # Segunda passada: agora sobre dados limpos.
    g = d.groupby(chaves, observed=True)[alvo]
    d["_mediana"] = g.transform("median")
    d["_n"] = g.transform("size")
    return d[d["_n"] >= min_medicoes], descartadas


def resumo_grupos(d: pd.DataFrame, alvo: str, chaves: list[str]) -> pd.DataFrame:
    """Uma linha por grupo, com a dispersão interna."""
    d = d.copy()
    d["_erro"] = (d[alvo] - d["_mediana"]).abs() / d["_mediana"]
    r = d.groupby(chaves, observed=True).agg(
        n=(alvo, "size"),
        tipo=("dispositivo_tipo", "first"),
        mediana=(alvo, "median"),
        # dispersão média dentro do grupo: é o piso do MAPE daquele grupo
        piso=("_erro", "mean"),
        p90=("_erro", lambda e: e.quantile(0.90)),
        # razão entre o melhor e o pior tempo no mesmo hardware
        amplitude=(alvo, lambda v: v.max() / v.min()),
    ).reset_index()
    return r


def relatar(g: pd.DataFrame, rotulo: str, descartadas: int) -> float:
    """Imprime o piso de um alvo e devolve o valor headline (mediana)."""
    piso = g["piso"].median()
    print(f"\n  {rotulo}")
    if descartadas:
        print(f"    descartadas por render abortado:{descartadas:>8,d}"
              .replace(",", "."))
    print(f"    grupos:                        {len(g):>8,d}".replace(",", "."))
    print(f"    submissões por grupo (mediana):{g['n'].median():>8.0f}")
    print(f"    PISO DO MAPE (mediana):        {piso * 100:>7.2f} %")
    print(f"    ... média entre grupos:        {g['piso'].mean() * 100:>7.2f} %"
          f"   <- puxada pela cauda")
    print(f"    p90 do erro (mediana):         {g['p90'].median() * 100:>7.2f} %")
    print(f"    amplitude máx/mín (mediana):   {g['amplitude'].median():>7.2f} x")

    print(f"    por tipo de dispositivo:")
    por_tipo = (g.groupby("tipo")
                  .agg(grupos=("n", "size"), piso=("piso", "median"))
                  .sort_values("piso"))
    for tipo, linha in por_tipo.iterrows():
        print(f"      {tipo:8s} {int(linha['grupos']):>6,d} grupos   "
              f"piso {linha['piso'] * 100:>6.2f} %".replace(",", "."))
    return piso


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _salvar(fig, nome: str, dir_figuras: Path) -> None:
    caminho = dir_figuras / nome
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    {caminho.relative_to(RAIZ)}")


def fig_distribuicao_ruido(grupos: dict[str, pd.DataFrame],
                           dir_figuras: Path) -> None:
    """Histograma do piso de cada grupo, por alvo.

    Responde: o ruído é uma cauda de poucas máquinas problemáticas ou é
    difuso? A distância entre a mediana e a média mede a cauda.
    """
    fig, eixos = plt.subplots(1, len(grupos), figsize=(11, 4), sharey=True)
    eixos = np.atleast_1d(eixos)
    for ax, (alvo, g) in zip(eixos, grupos.items()):
        x = g["piso"] * 100
        ax.hist(x.clip(upper=300), bins=np.logspace(-1, 2.5, 60),
                color="#4C72B0", edgecolor="none")
        ax.axvline(x.median(), color="#C44E52", lw=2,
                   label=f"piso = {x.median():.1f} %")
        ax.axvline(x.mean(), color="#DD8452", lw=1.5, ls="--",
                   label=f"média = {x.mean():.1f} %")
        ax.set_xscale("log")
        ax.set_xlabel("dispersão interna do grupo (%)")
        ax.set_title(ALVOS[alvo])
        ax.legend(frameon=False, fontsize=9)
    eixos[0].set_ylabel("grupos")
    fig.suptitle("Erro irredutível: variação entre submissões do mesmo hardware",
                 fontsize=12)
    _salvar(fig, "erro_irredutivel_distribuicao.png", dir_figuras)


def fig_por_tipo(grupos: dict[str, pd.DataFrame], dir_figuras: Path) -> None:
    """Boxplot do ruído por tipo de dispositivo.

    Responde: o piso é o mesmo para CPU e GPU? Se não for, o modelo tem de
    ser avaliado contra pisos separados por tipo.
    """
    fig, eixos = plt.subplots(1, len(grupos), figsize=(11, 4.2), sharey=True)
    eixos = np.atleast_1d(eixos)
    for ax, (alvo, g) in zip(eixos, grupos.items()):
        ordem = g.groupby("tipo")["piso"].median().sort_values().index
        dados = [g.loc[g["tipo"] == t, "piso"] * 100 for t in ordem]
        bp = ax.boxplot(dados, tick_labels=list(ordem), showfliers=False,
                        patch_artist=True, medianprops={"color": "black"})
        for caixa, tipo in zip(bp["boxes"], ordem):
            caixa.set_facecolor(CORES.get(tipo, "#999999"))
            caixa.set_alpha(0.75)
        ax.set_yscale("log")
        ax.set_title(ALVOS[alvo])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    eixos[0].set_ylabel("dispersão interna do grupo (%)")
    fig.suptitle("Ruído de medição por tipo de dispositivo", fontsize=12)
    _salvar(fig, "erro_irredutivel_por_tipo.png", dir_figuras)


def fig_ruido_vs_submissoes(grupos: dict[str, pd.DataFrame],
                            dir_figuras: Path) -> None:
    """Ruído em função de quantas vezes a máquina submeteu.

    É a figura que informa a escolha do --min-medicoes em
    montar_dataset_opendata.py: máquinas que submetem muito são mais
    ruidosas (mais condições diferentes ao longo dos anos) ou menos?
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    faixas = [1, 2, 3, 5, 10, 20, 30, 50, 100, 300, 1000, 10 ** 9]
    rotulos = ["1", "2", "3-4", "5-9", "10-19", "20-29",
               "30-49", "50-99", "100-299", "300-999", "1000+"]
    for (alvo, g), cor in zip(grupos.items(), ("#4C72B0", "#DD8452")):
        faixa = pd.cut(g["n"], bins=faixas, right=False, labels=rotulos)
        m = g.groupby(faixa, observed=False)["piso"].median() * 100
        ax.plot(range(len(rotulos)), m.reindex(rotulos).to_numpy(),
                "o-", color=cor, label=ALVOS[alvo])
    ax.set_xticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, rotation=45)
    ax.set_xlabel("submissões do mesmo grupo")
    ax.set_ylabel("dispersão interna, mediana (%)")
    ax.set_title("O limiar --min-medicoes seleciona máquinas mais ruidosas?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    _salvar(fig, "erro_irredutivel_vs_submissoes.png", dir_figuras)


def fig_piso_vs_alvo(grupos: dict[str, pd.DataFrame], dir_figuras: Path) -> None:
    """Piso do erro contra a magnitude do alvo.

    Responde se o ruído é multiplicativo (constante em termos relativos:
    linha horizontal) ou aditivo (cai com a magnitude). Isso decide se o
    modelo deve ser treinado no log do alvo -- é o que justifica as colunas
    log_vazao e log_custo_fixo do dataset.
    """
    fig, eixos = plt.subplots(1, len(grupos), figsize=(11, 4.2))
    eixos = np.atleast_1d(eixos)
    for ax, (alvo, g) in zip(eixos, grupos.items()):
        x = g["mediana"] / (1e6 if alvo.startswith("vazao") else 1.0)
        for tipo in g["tipo"].unique():
            m = g["tipo"] == tipo
            ax.scatter(x[m], g.loc[m, "piso"] * 100, s=5, alpha=0.35,
                       color=CORES.get(tipo, "#999999"), label=tipo,
                       edgecolors="none")
        # tendência: mediana do piso por decil da magnitude do alvo
        decil = pd.qcut(x, 10, duplicates="drop")
        t = g.groupby(decil, observed=True)["piso"].median() * 100
        ax.plot([i.mid for i in t.index], t.to_numpy(), "k-", lw=2,
                label="mediana por decil")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("vazão do grupo (Mamostra·px/s)" if alvo.startswith("vazao")
                      else "custo fixo do grupo (s)")
        ax.set_title(ALVOS[alvo])
        ax.grid(alpha=0.3)
    eixos[0].set_ylabel("piso do erro (%)")
    eixos[-1].legend(frameon=False, fontsize=8, markerscale=2)
    fig.suptitle("O ruído é relativo ou absoluto? (decide o treino em log)",
                 fontsize=12)
    _salvar(fig, "erro_irredutivel_vs_alvo.png", dir_figuras)


def fig_vies_popularidade(df: pd.DataFrame, dir_figuras: Path) -> None:
    """Participação de cada tipo entre máquinas vs entre submissões.

    É a figura que justifica a agregação por mediana: treinar sem agregar
    seria treinar numa população de hardware que não existe.
    """
    por_maquina = (df.drop_duplicates("maquina")["dispositivo_tipo"]
                     .value_counts(normalize=True) * 100)
    por_medicao = df["dispositivo_tipo"].value_counts(normalize=True) * 100
    ordem = por_maquina.index

    x = np.arange(len(ordem))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - 0.2, por_maquina[ordem], 0.4, label="% das máquinas",
           color="#4C72B0")
    ax.bar(x + 0.2, por_medicao[ordem], 0.4, label="% das submissões",
           color="#C44E52")
    for i, t in enumerate(ordem):
        razao = por_medicao[t] / por_maquina[t]
        ax.text(i, max(por_maquina[t], por_medicao[t]) + 1.5,
                f"{razao:.1f}×", ha="center", fontsize=9,
                color="#C44E52" if razao > 1 else "#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(ordem, rotation=45)
    ax.set_ylabel("participação (%)")
    ax.set_ylim(0, max(por_maquina.max(), por_medicao.max()) * 1.15)
    ax.set_title("Viés de popularidade do Open Data\n"
                 "(× = quantas vezes o tipo é super/sub-representado)",
                 fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _salvar(fig, "vies_popularidade.png", dir_figuras)


def fig_alvos_por_cena(df: pd.DataFrame, dir_figuras: Path) -> None:
    """Distribuição dos dois alvos por cena, um voto por máquina.

    Mostra a separação entre cenas que o modelo precisa aprender, e por que
    o custo fixo merece um modelo próprio: a ordem das cenas nele é outra.
    """
    med = (df.groupby(["maquina", "cena"], observed=True)
             [["vazao_amostras_pixel_s", "custo_fixo_s"]]
             .median().reset_index())

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2))
    cenas = sorted(med["cena"].unique())
    for ax, alvo in zip(eixos, ALVOS):
        dados = [med.loc[med["cena"] == c, alvo].dropna() for c in cenas]
        if alvo == "vazao_amostras_pixel_s":
            dados = [d / 1e6 for d in dados]
        bp = ax.boxplot(dados, tick_labels=cenas, showfliers=False,
                        patch_artist=True, medianprops={"color": "black"})
        for caixa in bp["boxes"]:
            caixa.set_facecolor("#55A868")
            caixa.set_alpha(0.7)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    eixos[0].set_ylabel("vazão (Mamostra·px/s)")
    eixos[1].set_ylabel("custo fixo (s)")
    eixos[0].set_title("Alvo 1: vazão")
    eixos[1].set_title("Alvo 2: custo fixo")
    fig.suptitle("Os dois alvos por cena (um voto por máquina)", fontsize=12)
    _salvar(fig, "alvos_por_cena.png", dir_figuras)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=CSV_PADRAO,
                    help="dataset NÃO agregado (--agregar nenhum)")
    ap.add_argument("--min-medicoes", type=int, default=5,
                    help="mínimo de submissões por grupo para ele entrar na "
                         "estimativa (padrão: 5)")
    ap.add_argument("--corte-outlier", type=float, default=10.0,
                    help="descarta submissões a mais de N vezes da mediana do "
                         "grupo -- renders abortados (padrão: 10)")
    ap.add_argument("--sem-versao", action="store_true",
                    help="não separa grupos por versão maior do Blender; o "
                         "piso passa a incluir a evolução do renderizador")
    ap.add_argument("--figuras", type=Path, default=DIR_FIGURAS)
    ap.add_argument("--sem-figuras", action="store_true")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"Não encontrei {args.csv}.\n"
            f"Gere com: python scripts/montar_dataset_opendata.py "
            f"--agregar nenhum --saida {args.csv.relative_to(RAIZ)}")

    df = carregar(args.csv)

    chaves = ["maquina", "cena"]
    if not args.sem_versao:
        chaves.append("versao_blender_major")

    print(f"\n{'=' * 62}")
    print("ERRO IRREDUTÍVEL")
    print(f"  grupo         = {' x '.join(chaves)}")
    print(f"  min-medicoes  = {args.min_medicoes}")
    print(f"  corte-outlier = {args.corte_outlier:g}x a mediana do grupo")
    print("=" * 62)

    grupos: dict[str, pd.DataFrame] = {}
    for alvo in ALVOS:
        d, descartadas = preparar(df, alvo, chaves,
                                  args.min_medicoes, args.corte_outlier)
        g = resumo_grupos(d, alvo, chaves)
        grupos[alvo] = g
        relatar(g, ALVOS[alvo].upper(), descartadas)

    print(f"\n{'=' * 62}")
    print("Leitura: nenhum modelo pode ficar abaixo desses MAPEs, porque as")
    print("features são idênticas nas submissões que estão sendo comparadas.")
    print("=" * 62)

    if not args.sem_figuras:
        args.figuras.mkdir(parents=True, exist_ok=True)
        print("\nfiguras:")
        fig_distribuicao_ruido(grupos, args.figuras)
        fig_por_tipo(grupos, args.figuras)
        fig_ruido_vs_submissoes(grupos, args.figuras)
        fig_piso_vs_alvo(grupos, args.figuras)
        fig_vies_popularidade(df, args.figuras)
        fig_alvos_por_cena(df, args.figuras)


if __name__ == "__main__":
    main()
