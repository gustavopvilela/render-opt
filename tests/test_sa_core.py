from dataclasses import dataclass
import numpy as np
import pytest

sa_core = pytest.importorskip("sa_core")

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

def test_tempos_batem_com_python():
    inst = instancia_aleatoria()
    solucao = solucao_aleatoria(inst)
    esperado = [tempo_maquina(s, k, inst) for k, s in enumerate(solucao)]
    np.testing.assert_allclose(sa_core.calcular_tempos_maquinas(inst, solucao), esperado)

def test_solucao_valida_e_makespan_consistente ():
    inst = instancia_aleatoria()
    solucao, makespan, _ = sa_core.simulated_annealing(inst, semente=1, verbose=False)

    assert len(solucao) == inst.maquinas
    assert sorted(j for seq in solucao for j in seq) == list(range(inst.jobs))
    tempos = sa_core.calcular_tempos_maquinas(inst, solucao)
    assert makespan == pytest.approx(tempos.max())

def test_melhora_a_solucao_inicial ():
    inst = instancia_aleatoria()
    inicial = solucao_aleatoria(inst)
    makespan_inicial = sa_core.calcular_tempos_maquinas(inst, inicial).max()
    _, makespan, _ = sa_core.simulated_annealing(inst, semente=1, solucao_inicial=inicial, verbose=False)
    assert makespan < makespan_inicial

def test_mesma_semente_mesmo_resultado ():
    inst = instancia_aleatoria()
    a = sa_core.simulated_annealing(inst, semente=42, verbose=False)
    b = sa_core.simulated_annealing(inst, semente=42, verbose=False)
    assert a[0] == b[0] and a[1] == b[1]

def test_historico ():
    inst = instancia_aleatoria()
    _, makespan, hist = sa_core.simulated_annealing(inst, semente=1, verbose=False)
    melhor = hist["melhor_makespan"]
    assert set(hist) == {"iteracao", "temperatura", "makespan_atual", "melhor_makespan"}
    assert len(melhor) > 0
    assert np.all(np.diff(melhor) <= 0)
    assert np.all(hist["makespan_atual"] >= melhor)
    assert melhor[-1] == pytest.approx(makespan)

def test_criterios_de_parada ():
    inst = instancia_aleatoria()
    _, _, hist = sa_core.simulated_annealing(inst, semente=0, max_iteracoes_sem_melhora=50, verbose=False)
    assert len(hist["iteracao"]) < 68_800

def test_erros_viram_valueerror ():
    inst = instancia_aleatoria()
    inst.matriz_setup = inst.matriz_setup[:, :, :-1]
    with pytest.raises(ValueError, match="matriz_setup"):
        sa_core.simulated_annealing(inst, verbose=False)

    inst = instancia_aleatoria()
    with pytest.raises(ValueError, match="aparece mais de uma vez"):
        sa_core.simulated_annealing(inst, solucao_inicial=[[0, 0]] + [[]] * 3, verbose=False)