"""
Distribuição do Flamenco em instâncias de Vallada e Ruiz (2011): Gantt e resultados.csv.

    python experiments/flamenco/simular_instancias.py experiments/flamenco/instancias --sa --semente 1
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sa_core
from master import flamenco

AQUI = Path(__file__).resolve().parent
COR = "#2a78d6"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRADE = "#e4e3df"

def carregar_instancia (caminho: Path) -> SimpleNamespace:
    linhas = [l.split() for l in caminho.read_text().splitlines() if l.strip()]
    jobs, maquinas = int(linhas[0][0]), int(linhas[0][1])
    P = np.zeros((jobs, maquinas))
    for j, linha in enumerate(linhas[2:2 + jobs]):
        pares = np.array(linha, dtype=float).reshape(-1, 2)
        P[j, pares[:, 0].astype(int)] = pares[:, 1]
    i_ssd = next(i for i, l in enumerate(linhas) if l[0] == "SSD")
    S = np.zeros((maquinas, jobs, jobs))
    for k in range(maquinas):
        inicio = i_ssd + 1 + k * (jobs + 1)
        if linhas[inicio][0] != f"M{k}":
            raise ValueError(f"Esperado M{k} em {caminho}")
        S[k] = np.array(linhas[inicio + 1:inicio + 1 + jobs], dtype=float)
    return SimpleNamespace(jobs=jobs, maquinas=maquinas, matriz_processamento=P, matriz_setup=S)

def cronograma (inst, plano: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(início do setup, início do processamento, fim) de cada job."""
    inicio, processamento, fim = (np.zeros(inst.jobs) for _ in range(3))
    for k, seq in enumerate(plano):
        t, anterior = 0.0, -1
        for j in seq:
            inicio[j] = t
            t += inst.matriz_setup[k, anterior, j] if anterior >= 0 else 0.0
            processamento[j] = t
            t += inst.matriz_processamento[j, k]
            fim[j], anterior = t, j
    return inicio, processamento, fim

def desenhar_gantt (eixo, inst, plano, inicio, processamento, fim, makespan: float, titulo: str) -> None:
    rotular = inst.jobs <= 60
    for k, seq in enumerate(plano):
        for j in seq:
            if processamento[j] > inicio[j]:
                eixo.barh(k, processamento[j] - inicio[j], left=inicio[j], height=0.62, color="white",
                          edgecolor=COR, hatch="////", linewidth=0.6)
            eixo.barh(k, fim[j] - processamento[j], left=processamento[j], height=0.62, color=COR,
                      edgecolor="white", linewidth=1.0)
            if rotular and fim[j] - processamento[j] > makespan / 40:
                eixo.text((processamento[j] + fim[j]) / 2, k, str(j), ha="center", va="center", fontsize=6, color="white")
    eixo.axvline(makespan, color=TEXTO, linestyle="--", linewidth=1.0)
    eixo.text(makespan, inst.maquinas - 0.4, f" Cmax = {makespan:g}", va="bottom", ha="left", fontsize=8, color=TEXTO)
    eixo.set_yticks(range(inst.maquinas), [f"M{k}" for k in range(inst.maquinas)], fontsize=7 if inst.maquinas > 15 else 8)
    eixo.invert_yaxis()
    eixo.set_title(titulo, loc="left", fontsize=10, color=TEXTO)
    eixo.grid(axis="x", color=GRADE, linewidth=0.6)
    eixo.set_axisbelow(True)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    eixo.tick_params(colors=TEXTO_SECUNDARIO)

def salvar_gantt (destino: Path, nome: str, inst, paineis: list[tuple]) -> None:
    altura = max(2.2, 0.22 * inst.maquinas + 1.0)
    fig, eixos = plt.subplots(len(paineis), 1, figsize=(10, altura * len(paineis) + 0.6), sharex=True, squeeze=False)
    for eixo, painel in zip(eixos[:, 0], paineis):
        desenhar_gantt(eixo, inst, *painel)
    eixos[-1, 0].set_xlabel("Tempo", color=TEXTO_SECUNDARIO)
    setup = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=COR, hatch="////")
    proc = plt.Rectangle((0, 0), 1, 1, facecolor=COR)
    fig.legend([proc, setup], ["Processamento", "Setup"], loc="upper right", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"{nome} ({inst.jobs} jobs, {inst.maquinas} máquinas)", x=0.01, ha="left", fontsize=11, color=TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destino, dpi=200)
    plt.close(fig)

def salvar_razoes (destino: Path, linhas: list[dict]) -> None:
    nomes = [l["instancia"] for l in linhas]
    razoes = [l["razao_flamenco_sa"] for l in linhas]
    fig, eixo = plt.subplots(figsize=(8, max(2.5, 0.28 * len(nomes) + 1.0)))
    eixo.axvline(1.0, color=TEXTO_SECUNDARIO, linewidth=1.0)
    eixo.hlines(range(len(nomes)), 1.0, razoes, color=COR, linewidth=2)
    eixo.scatter(razoes, range(len(nomes)), color=COR, s=36, zorder=3, edgecolors="white", linewidths=1.5)
    for i, r in enumerate(razoes):
        eixo.text(r, i, f"  {r:.2f}", va="center", fontsize=7, color=TEXTO)
    eixo.set_yticks(range(len(nomes)), nomes, fontsize=7)
    eixo.invert_yaxis()
    eixo.set_xlabel("Cmax(Flamenco) / Cmax(SA)", color=TEXTO_SECUNDARIO)
    eixo.set_title("Quanto a ordem de chegada fica acima do SA (1 = empate)", loc="left", fontsize=10, color=TEXTO)
    eixo.grid(axis="x", color=GRADE, linewidth=0.6)
    eixo.set_axisbelow(True)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)

def instancias_de (caminhos: list[Path]) -> list[Path]:
    arquivos = []
    for c in caminhos:
        arquivos += sorted(c.glob("*.txt")) if c.is_dir() else [c]
    if not arquivos:
        raise SystemExit("Nenhuma instância encontrada")
    return arquivos

def main (argv=None) -> int:
    p = argparse.ArgumentParser(description="Simula a distribuição do Flamenco em instâncias de Vallada e Ruiz.")
    p.add_argument("caminhos", type=Path, nargs="+", help="Arquivos .txt ou pastas com eles")
    p.add_argument("--saida", type=Path, default=AQUI / "resultados")
    p.add_argument("--chunk", type=int, default=flamenco.TAMANHO_CHUNK_PADRAO, help="Jobs por tarefa (chunk_size do Flamenco)")
    p.add_argument("--esperas-do-flamenco", action="store_true",
                   help="Soma as esperas do trabalhador do Flamenco (3 s e 5 s) na unidade de tempo da instância")
    p.add_argument("--sa", action="store_true", help="Roda também o SA de Santos et al. (tempo limite do artigo)")
    p.add_argument("--semente", type=int, default=None, help="Semente do SA")
    p.add_argument("--max-iteracoes", type=int, default=None, help="Para o SA por iterações em vez do tempo do artigo")
    p.add_argument("--sem-graficos", action="store_true")
    a = p.parse_args(argv)

    a.saida.mkdir(parents=True, exist_ok=True)
    esperas = {}
    if a.esperas_do_flamenco:
        esperas = {"espera_inicial": flamenco.ESPERA_INICIAL_S, "intervalo_entre_tarefas": flamenco.INTERVALO_ENTRE_TAREFAS_S}

    linhas = []
    for caminho in instancias_de(a.caminhos):
        inst = carregar_instancia(caminho)
        nome = caminho.stem
        sim = flamenco.simular(inst, tarefas=flamenco.dividir_em_tarefas([inst.jobs], a.chunk), **esperas)
        linha = {"instancia": nome, "jobs": inst.jobs, "maquinas": inst.maquinas, "makespan_flamenco": sim.makespan}
        paineis = [(sim.plano, sim.inicio, sim.inicio_processamento, sim.fim, sim.makespan, "Flamenco (ordem de chegada)")]

        if a.sa:
            t0 = time.perf_counter()
            plano, makespan, _ = sa_core.simulated_annealing(inst, semente=a.semente, max_iteracoes=a.max_iteracoes, verbose=False)
            linha |= {"makespan_sa": makespan, "razao_flamenco_sa": sim.makespan / makespan,
                      "tempo_sa_s": time.perf_counter() - t0}
            paineis.append((plano, *cronograma(inst, plano), makespan, "Simulated Annealing (Santos et al., 2016)"))

        linhas.append(linha)
        print(", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in linha.items()))
        if not a.sem_graficos:
            salvar_gantt(a.saida / f"gantt_{nome}.png", nome, inst, paineis)

    with (a.saida / "resultados.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)

    if a.sa and len(linhas) > 1 and not a.sem_graficos:
        salvar_razoes(a.saida / "razao_flamenco_sa.png", linhas)
    print(f"Resultados em {a.saida}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
