import argparse
import sys
from types import SimpleNamespace
import numpy as np
import sa_core

ORCAMENTO_ITERACOES = 600_000

def carregar_instancia (caminho):
    with open(caminho) as f:
        linhas = [l.split() for l in f if l.strip()]

    jobs, maquinas = int(linhas[0][0]), int(linhas[0][1])
    P = np.zeros((jobs, maquinas))
    for j, linha in enumerate(linhas[2:2 + jobs]):
        pares = np.array(linha, dtype=float).reshape(-1, 2)
        P[j, pares[:, 0].astype(int)] = pares[:, 1]

    i_ssd = next(i for i, l in enumerate(linhas) if l[0] == "SSD")
    S = np.zeros((maquinas, jobs, jobs))
    for k in range(maquinas):
        inicio = i_ssd + 1 + k * (jobs + 1)
        assert linhas[inicio][0] == f"M{k}", f"Esperado M{k} em {caminho}"
        S[k] = np.array(linhas[inicio + 1:inicio + 1 + jobs], dtype=float)

    return SimpleNamespace(jobs=jobs, maquinas=maquinas, matriz_processamento=P, matriz_setup=S)

def limite_inferior (P):
    menor_por_job = P.min(axis=1)
    return max(menor_por_job.max(), menor_por_job.sum() / P.shape[1])

def main ():
    _id_config, _id_instancia, semente, caminho, *resto = sys.argv[1:]

    p = argparse.ArgumentParser()
    p.add_argument("--fator_t0", type=float, required=True)
    p.add_argument("--fator_tf", type=float, required=True)
    p.add_argument("--iter_por_temp", type=int, required=True)
    p.add_argument("--taxa", type=float, required=True)
    p.add_argument("--alpha_construcao", type=float, required=True)
    a = p.parse_args(resto)

    inst = carregar_instancia(caminho)
    escala = limite_inferior(inst.matriz_processamento)
    T0 = a.fator_t0 * escala
    Tf = a.fator_tf * escala
    alfa = (Tf / T0) ** (a.iter_por_temp / ORCAMENTO_ITERACOES)

    _, makespan, _ = sa_core.simulated_annealing(
        inst, T0, alfa, a.iter_por_temp,
        temp_final=Tf,
        taxa=a.taxa,
        alpha_construcao=a.alpha_construcao,
        semente=int(semente),
        intervalo_historico=ORCAMENTO_ITERACOES,
        verbose=False
    )
    print(makespan)

if __name__ == "__main__":
    main()