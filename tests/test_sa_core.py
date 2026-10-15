import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sa_core = pytest.importorskip("sa_core")

INSTANCIAS_TESTE = Path(__file__).resolve().parent.parent / "experiments" / "irace" / "instancias" / "teste"

@dataclass
class UPMSPInstance:
    jobs: int
    maquinas: int
    matriz_processamento: np.ndarray
    matriz_setup: np.ndarray

def instancia_aleatoria (jobs=40, maquinas=4, seed=0):
    rng = np.random.default_rng(seed)

    return UPMSPInstance(
        jobs=jobs,
        maquinas=maquinas,
        matriz_processamento=rng.uniform(1, 100, size=(jobs, maquinas)),
        matriz_setup=rng.uniform(0, 10, size=(maquinas, jobs, jobs))
    )

def tempo_maquina (sequencia, maquina, inst):
    tempo = 0.0
    for pos, job in enumerate(sequencia):
        if pos > 0:
            tempo += inst.matriz_setup[maquina, sequencia[pos - 1], job]
        tempo += inst.matriz_processamento[job, maquina]
    return tempo

def solucao_aleatoria(inst, seed=0):
    rng = np.random.default_rng(seed)
    solucao = [[] for _ in range(inst.maquinas)]
    for job in rng.permutation(inst.jobs):
        solucao[rng.integers(inst.maquinas)].append(int(job))
    return solucao

def carregar_vallada (caminho):
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
        S[k] = np.array(linhas[inicio + 1:inicio + 1 + jobs], dtype=float)
    return SimpleNamespace(jobs=jobs, maquinas=maquinas, matriz_processamento=P, matriz_setup=S)

def sa (inst, max_iteracoes=20_000, **kwargs):
    return sa_core.simulated_annealing(inst, max_iteracoes=max_iteracoes, verbose=False, **kwargs)

def test_tempos_batem_com_python():
    inst = instancia_aleatoria()
    solucao = solucao_aleatoria(inst)
    esperado = [tempo_maquina(s, k, inst) for k, s in enumerate(solucao)]
    np.testing.assert_allclose(sa_core.calcular_tempos_maquinas(inst, solucao), esperado)

def test_solucao_valida_e_makespan_consistente ():
    inst = instancia_aleatoria()
    solucao, makespan, _ = sa(inst, semente=1)

    assert len(solucao) == inst.maquinas
    assert sorted(j for seq in solucao for j in seq) == list(range(inst.jobs))
    tempos = sa_core.calcular_tempos_maquinas(inst, solucao)
    assert makespan == pytest.approx(tempos.max())

def test_melhora_a_solucao_inicial ():
    inst = instancia_aleatoria()
    inicial = solucao_aleatoria(inst)
    makespan_inicial = sa_core.calcular_tempos_maquinas(inst, inicial).max()
    _, makespan, _ = sa(inst, semente=1, solucao_inicial=inicial)
    assert makespan < makespan_inicial

def test_mesma_semente_mesmo_resultado ():
    inst = instancia_aleatoria()
    a = sa(inst, semente=42)
    b = sa(inst, semente=42)
    assert a[0] == b[0] and a[1] == b[1]

def test_historico ():
    inst = instancia_aleatoria()
    _, makespan, hist = sa(inst, semente=1)
    melhor = hist["melhor_makespan"]
    assert {"iteracao", "temperatura", "makespan_atual", "melhor_makespan"} <= set(hist)
    assert len(melhor) > 1
    assert np.all(np.diff(melhor) < 0)
    assert np.all(hist["makespan_atual"] >= melhor)
    assert melhor[-1] == pytest.approx(makespan)
    assert hist["iteracoes"] == 20_000

# Valores do código Java dos autores (novo-sa/upmsp), com o SA.run limitado por iterações em vez de tempo
@pytest.mark.parametrize("semente, t0, alpha, samax, makespan, reaquecimentos, solucao", [
    (0, 1.0, 0.96, 1_176_628, 73, 0, [
        [48, 31, 2, 0], [45, 7, 43], [46, 6, 24, 27, 8, 13], [41, 10, 28, 3, 18, 36], [5, 33, 44, 25],
        [14, 21, 29, 34, 35, 26, 42], [38, 4, 15, 20], [1, 37, 17, 32], [9, 49, 19, 12, 16], [47, 39, 30, 40, 23, 22, 11]]),
    (3, 100.0, 0.9, 100, 75, 11, [
        [2, 0, 28, 48, 7], [19, 45], [27, 8, 24, 13, 46, 31], [4, 41, 10, 3, 36, 6], [44, 17, 34, 5],
        [29, 14, 21, 35, 26], [15, 20, 25, 38], [42, 33, 1, 37, 32], [9, 12, 49, 43, 18, 16], [47, 39, 30, 40, 23, 22, 11]]),
])
def test_reproduz_o_codigo_java (semente, t0, alpha, samax, makespan, reaquecimentos, solucao):
    caminho = INSTANCIAS_TESTE / "I_50_10_S_1-9_1.txt"
    if not caminho.exists():
        pytest.skip("instância de Vallada e Ruiz ausente")
    inst = carregar_vallada(caminho)
    obtida, obtido, hist = sa_core.simulated_annealing(inst, t0, alpha, samax, max_iteracoes=200_000, semente=semente, verbose=False)
    assert obtida == solucao
    assert obtido == makespan
    assert hist["reaquecimentos"] == reaquecimentos

def test_construcao_aleatoria_reprodutivel ():
    inst = instancia_aleatoria()
    solucao = sa_core.gerar_solucao_aleatoria(inst, 5)
    assert sorted(j for seq in solucao for j in seq) == list(range(inst.jobs))
    assert solucao == sa_core.gerar_solucao_aleatoria(inst, 5)

def test_tempo_limite_artigo ():
    assert sa_core.tempo_limite_artigo(250, 30) == pytest.approx(75.0)
    assert sa_core.tempo_limite_artigo(6, 2) == pytest.approx(0.12)

def test_criterios_de_parada ():
    inst = instancia_aleatoria(jobs=10, maquinas=2)
    _, _, hist = sa_core.simulated_annealing(inst, max_iteracoes=1234, semente=0, verbose=False)
    assert hist["iteracoes"] == 1234

    _, _, hist = sa_core.simulated_annealing(inst, tempo_limite_segundos=0.05, semente=0, verbose=False)
    assert "tempo limite" in hist["motivo_parada"]
    assert 0.05 <= hist["tempo_segundos"] < 1.0

    _, _, hist = sa_core.simulated_annealing(inst, semente=0, verbose=False)
    assert hist["tempo_segundos"] >= sa_core.tempo_limite_artigo(10, 2)

def test_reaquecimento ():
    inst = instancia_aleatoria(jobs=10, maquinas=2)
    _, _, hist = sa_core.simulated_annealing(inst, 1.0, 0.5, 1, max_iteracoes=100, semente=0, verbose=False)
    assert hist["reaquecimentos"] == 5

def test_setup_inicial_e_tempos_reais ():
    inst = instancia_aleatoria(jobs=30, maquinas=3)
    inst.setup_inicial = np.random.default_rng(1).uniform(0, 50, size=(inst.maquinas, inst.jobs))
    solucao, makespan, hist = sa(inst, semente=3)
    esperado = max(
        inst.setup_inicial[k, seq[0]] + tempo_maquina(seq, k, inst) if seq else 0.0
        for k, seq in enumerate(solucao)
    )
    assert makespan == pytest.approx(esperado)
    assert hist["melhor_makespan"][-1] == pytest.approx(makespan)

def test_maquinas_inelegiveis ():
    inst = instancia_aleatoria(jobs=20, maquinas=4)
    inst.matriz_processamento[:10, 0] = math.inf
    inst.matriz_processamento[10:, 3] = math.inf
    solucao, makespan, hist = sa(inst, semente=0)
    assert math.isfinite(makespan)
    assert not any(j < 10 for j in solucao[0]) and not any(j >= 10 for j in solucao[3])
    assert hist["melhor_makespan"][-1] == pytest.approx(makespan)

def test_solucao_inicial_infinita_e_recusada ():
    inst = instancia_aleatoria(jobs=4, maquinas=2)
    inst.matriz_processamento[0, 1] = math.inf
    with pytest.raises(ValueError, match="infinito"):
        sa(inst, solucao_inicial=[[1, 2, 3], [0]])

@pytest.mark.parametrize("jobs, maquinas", [(0, 3), (1, 1), (5, 1), (3, 5), (1, 4)])
def test_casos_degenerados (jobs, maquinas):
    inst = instancia_aleatoria(jobs=jobs, maquinas=maquinas)
    solucao, makespan, _ = sa(inst, max_iteracoes=2_000, semente=0)
    assert sorted(j for seq in solucao for j in seq) == list(range(jobs))
    assert makespan == pytest.approx(sa_core.calcular_tempos_maquinas(inst, solucao).max())

def test_vizinhancas ():
    inst = instancia_aleatoria(jobs=12, maquinas=3)
    so_task_move = [False] * 24
    so_task_move[4 * 4 + 1] = True
    solucao, _, _ = sa(inst, semente=0, vizinhancas=so_task_move)
    assert sorted(j for seq in solucao for j in seq) == list(range(inst.jobs))

    with pytest.raises(ValueError, match="24"):
        sa(inst, vizinhancas=[True] * 23)
    with pytest.raises(ValueError, match="vizinhança"):
        sa(inst, vizinhancas=[False] * 24)

def test_erros_viram_valueerror ():
    inst = instancia_aleatoria()
    inst.matriz_setup = inst.matriz_setup[:, :, :-1]
    with pytest.raises(ValueError, match="matriz_setup"):
        sa(inst)

    inst = instancia_aleatoria()
    with pytest.raises(ValueError, match="aparece mais de uma vez"):
        sa(inst, solucao_inicial=[[0, 0]] + [[]] * 3)

    with pytest.raises(ValueError, match="resfriamento"):
        sa(inst, taxa_resfriamento=1.0)

def test_delta_insercao_bate_com_recalculo ():
    inst = instancia_aleatoria(jobs=12, maquinas=3)
    sequencia = [4, 0, 7, 2]
    for maquina in range(inst.maquinas):
        antes = tempo_maquina(sequencia, maquina, inst)
        for job in (1, 5, 11):
            for pos in range(len(sequencia) + 1):
                nova = sequencia[:pos] + [job] + sequencia[pos:]
                esperado = tempo_maquina(nova, maquina, inst) - antes
                obtido = sa_core.calcular_delta_insercao(job, maquina, pos, sequencia, inst)
                assert obtido == pytest.approx(esperado)

    assert sa_core.calcular_delta_insercao(3, 0, 0, [], inst) == pytest.approx(inst.matriz_processamento[3, 0])

@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test_solucao_inicial_valida(alpha):
    inst = instancia_aleatoria()
    for semente in range(5):
        solucao = sa_core.gerar_solucao_inicial(inst, alpha, semente=semente)
        assert len(solucao) == inst.maquinas
        assert sorted(j for seq in solucao for j in seq) == list(range(inst.jobs))

def test_solucao_inicial_reprodutivel ():
    inst = instancia_aleatoria()
    assert sa_core.gerar_solucao_inicial(inst, semente=7) == sa_core.gerar_solucao_inicial(inst, semente=7)

def test_alpha_constrola_gulosidade ():
    inst = instancia_aleatoria()

    def makespan_medio (alpha):
        return np.mean([
            sa_core.calcular_tempos_maquinas(inst, sa_core.gerar_solucao_inicial(inst, alpha, semente=s)).max() for s in range(20)
        ])

    assert makespan_medio(0.0) < makespan_medio(1.0)

def test_alpha_invalido ():
    inst = instancia_aleatoria()
    with pytest.raises(ValueError, match="Alpha"):
        sa_core.gerar_solucao_inicial(inst, alpha=1.5)