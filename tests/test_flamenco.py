import heapq
import math
import sqlite3
from types import SimpleNamespace
import numpy as np
import pytest

sa_core = pytest.importorskip("sa_core")

from fastapi.testclient import TestClient
from comum.util import sha256_arquivo
from master import flamenco
from master.app import criar_app
from master.database import DB
from test_master_worker import cfg_teste, concluir, registrar
from test_sa_core import INSTANCIAS_TESTE, carregar_vallada

def instancia (jobs=30, maquinas=4, seed=0, inelegiveis=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(1, 100, size=(jobs, maquinas))
    S = rng.uniform(0, 10, size=(maquinas, jobs, jobs))
    for j in rng.choice(jobs, size=inelegiveis, replace=False):
        P[j, rng.choice(maquinas, size=maquinas - 1, replace=False)] = math.inf
    return SimpleNamespace(jobs=jobs, maquinas=maquinas, matriz_processamento=P, matriz_setup=S)

def lista_de_graham (inst, tarefas, espera=0.0, intervalo=0.0):
    P, S = inst.matriz_processamento, inst.matriz_setup
    fila = [(espera, k) for k in range(inst.maquinas)]
    ultimo = [-1] * inst.maquinas
    plano = [[] for _ in range(inst.maquinas)]
    fim = np.zeros(inst.jobs)
    pendentes = list(tarefas)
    while pendentes and fila:
        livre, k = heapq.heappop(fila)
        for tarefa in pendentes:
            if all(np.isfinite(P[j, k]) for j in tarefa):
                break
        else:
            continue
        pendentes.remove(tarefa)
        t = livre
        for j in tarefa:
            t += (S[k, ultimo[k], j] if ultimo[k] >= 0 else 0.0) + P[j, k]
            fim[j] = t
            plano[k].append(j)
            ultimo[k] = j
        heapq.heappush(fila, (t + intervalo, k))
    return plano, fim

@pytest.mark.parametrize("seed", range(5))
def test_igual_a_referencia_em_python (seed):
    inst = instancia(seed=seed, inelegiveis=5)
    tarefas = flamenco.dividir_em_tarefas([inst.jobs], 1)
    plano, makespan, d = sa_core.flamenco(inst)
    esperado, fim = lista_de_graham(inst, tarefas)
    assert plano == esperado
    np.testing.assert_allclose(d["fim"], fim)
    assert makespan == pytest.approx(fim.max())

def test_tempos_batem_com_o_sa_core ():
    inst = instancia(seed=1)
    plano, makespan, d = sa_core.flamenco(inst)
    tempos = sa_core.calcular_tempos_maquinas(inst, plano)
    np.testing.assert_allclose(d["tempos_maquinas"], tempos)
    assert makespan == pytest.approx(tempos.max())
    assert all(d["maquina"][j] == k for k, seq in enumerate(plano) for j in seq)

def test_maquina_livre_pega_o_proximo_da_lista ():
    P = np.array([[3., 6.], [2., 4.], [5., 5.], [1., 1.]])
    inst = SimpleNamespace(jobs=4, maquinas=2, matriz_processamento=P, matriz_setup=np.zeros((2, 4, 4)))
    plano, makespan, d = sa_core.flamenco(inst)
    assert plano == [[0, 2], [1, 3]]
    assert makespan == 8.0
    np.testing.assert_allclose(d["inicio"], [0, 0, 3, 4])

def test_respeita_elegibilidade_e_recusa_job_sem_maquina ():
    P = np.array([[1., math.inf], [1., math.inf], [math.inf, 1.]])
    inst = SimpleNamespace(jobs=3, maquinas=2, matriz_processamento=P, matriz_setup=np.zeros((2, 3, 3)))
    plano, _, _ = sa_core.flamenco(inst)
    assert plano == [[0, 1], [2]]

    inst.matriz_processamento = np.array([[1., 1.], [math.inf, math.inf], [1., 1.]])
    with pytest.raises(ValueError, match="nenhuma máquina elegível"):
        sa_core.flamenco(inst)

def test_chunks_vao_inteiros_para_uma_maquina ():
    inst = instancia(jobs=10, maquinas=3, seed=2)
    tarefas = flamenco.dividir_em_tarefas([inst.jobs], 3)
    assert tarefas == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    plano, _, d = sa_core.flamenco(inst, tarefas=tarefas)
    for tarefa in tarefas:
        assert len({d["maquina"][j] for j in tarefa}) == 1
    esperado, fim = lista_de_graham(inst, tarefas)
    assert plano == esperado
    np.testing.assert_allclose(d["fim"], fim)

def test_chunks_nao_misturam_cenas ():
    assert flamenco.dividir_em_tarefas([3, 2], 2) == [[0, 1], [2], [3, 4]]
    with pytest.raises(ValueError):
        flamenco.dividir_em_tarefas([3], 0)

def test_esperas_do_trabalhador_do_flamenco ():
    inst = instancia(jobs=12, maquinas=3, seed=3)
    tarefas = flamenco.dividir_em_tarefas([inst.jobs])
    sim = flamenco.simular(inst, espera_inicial=flamenco.ESPERA_INICIAL_S,
                           intervalo_entre_tarefas=flamenco.INTERVALO_ENTRE_TAREFAS_S)
    plano, fim = lista_de_graham(inst, tarefas, flamenco.ESPERA_INICIAL_S, flamenco.INTERVALO_ENTRE_TAREFAS_S)
    assert sim.plano == plano
    np.testing.assert_allclose(sim.fim, fim)
    assert sim.inicio.min() == pytest.approx(flamenco.ESPERA_INICIAL_S)
    assert sim.makespan > flamenco.simular(inst).makespan

def test_farm_paga_o_custo_fixo_em_todo_frame ():
    P = np.array([[10., 20.], [10., 20.], [10., 20.]])
    C = np.array([[2., 1.], [2., 1.], [2., 1.]])
    sim = flamenco.simular_farm(P, C)
    np.testing.assert_allclose(sim.inicio_processamento - sim.inicio, C[np.arange(3), sim.maquina])
    np.testing.assert_allclose(sim.fim - sim.inicio_processamento, P[np.arange(3), sim.maquina])

@pytest.mark.parametrize("kwargs, mensagem", [
    ({"tarefas": [[0, 1], [1]]}, "mais de uma vez"),
    ({"tarefas": [[0]]}, "não está em nenhuma tarefa"),
    ({"tarefas": [[0], []]}, "pelo menos um job"),
    ({"espera_inicial": -1.0}, ">= 0"),
    ({"intervalo_entre_tarefas": math.inf}, ">= 0"),
])
def test_valida_entrada (kwargs, mensagem):
    inst = instancia(jobs=2, maquinas=2)
    with pytest.raises(ValueError, match=mensagem):
        sa_core.flamenco(inst, **kwargs)

def test_instancia_de_vallada_e_ruiz ():
    caminho = INSTANCIAS_TESTE / "I_50_10_S_1-9_1.txt"
    if not caminho.exists():
        pytest.skip("instância de Vallada e Ruiz ausente")
    inst = carregar_vallada(caminho)
    plano, makespan, _ = sa_core.flamenco(inst)
    assert sorted(j for seq in plano for j in seq) == list(range(inst.jobs))
    assert makespan == pytest.approx(sa_core.calcular_tempos_maquinas(inst, plano).max())


def lote_flamenco (cliente, cfg, frames, lote="F1", versao="5.1"):
    cena = cfg.pasta_cenas / "cena.blend"
    cena.parent.mkdir(parents=True, exist_ok=True)
    cena.write_bytes(b"BLENDER-v501 conteudo falso")
    r = cliente.post("/lotes", json={"lote_id": lote, "distribuicao": "flamenco", "cenas": [{
        "cena_id": "cena", "cena_sha256": sha256_arquivo(cena), "versao_cena": versao, "frames": list(frames)}]})
    assert r.status_code == 200, r.text
    return r.json()

def pedir (cliente, tid):
    r = cliente.post(f"/trabalhadores/{tid}/proximo-job")
    return r.json() if r.status_code == 200 else None

def test_lote_flamenco_entrega_na_ordem_de_submissao_sem_teto (tmp_path):
    cfg = cfg_teste(tmp_path)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "rapida", fator=3.0)
        registrar(c, "lenta", fator=1.0)
        resumo = lote_flamenco(c, cfg, frames=range(1, 7))
        assert resumo["distribuicao"] == "flamenco"
        assert c.get("/lotes/F1").json()["por_maquina"] == {"(sem dono)": {"na fila": 6}}

        a, b = pedir(c, "rapida"), pedir(c, "lenta")
        assert (a["frame"], b["frame"]) == (1, 2)
        assert a["checkpoint_s"] is None and a["teto_s"] is None
        concluir(c, a, "rapida")
        a = pedir(c, "rapida")
        assert a["frame"] == 3
        concluir(c, a, "rapida")
        concluir(c, b, "lenta")

        while entregues := [(tid, j) for tid in ("rapida", "lenta") if (j := pedir(c, tid))]:
            for tid, j in entregues:
                concluir(c, j, tid)
        status = c.get("/lotes/F1").json()
        assert status["jobs_por_estado"] == {"CONCLUIDO": 6} and status["concluido"]

def test_lote_flamenco_falha_volta_no_mesmo_lugar_e_outra_maquina_pega (tmp_path):
    cfg = cfg_teste(tmp_path)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "a")
        registrar(c, "b")
        lote_flamenco(c, cfg, frames=[1, 2, 3])
        concluir(c, pedir(c, "a"), "a", status="falha")
        assert pedir(c, "a")["frame"] == 2
        assert pedir(c, "b")["frame"] == 1

def test_lote_flamenco_unica_maquina_tenta_de_novo (tmp_path):
    cfg = cfg_teste(tmp_path)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "a")
        lote_flamenco(c, cfg, frames=[1, 2])
        concluir(c, pedir(c, "a"), "a", status="falha")
        assert pedir(c, "a")["frame"] == 1

def test_lote_flamenco_respeita_a_versao_do_blender (tmp_path):
    cfg = cfg_teste(tmp_path)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "antiga", versoes=("4.5",))
        registrar(c, "nova", versoes=("5.1",))
        lote_flamenco(c, cfg, frames=[1, 2])
        assert pedir(c, "antiga") is None
        assert pedir(c, "nova")["frame"] == 1

def test_banco_antigo_ganha_a_coluna_distribuicao (tmp_path):
    caminho = tmp_path / "estado.sqlite3"
    conexao = sqlite3.connect(caminho)
    conexao.execute("CREATE TABLE lotes (id TEXT PRIMARY KEY, criado_em REAL NOT NULL, inicio REAL, fim REAL)")
    conexao.execute("INSERT INTO lotes (id, criado_em) VALUES ('L0', 0)")
    conexao.commit()
    conexao.close()
    db = DB(caminho)
    assert db.conexao.execute("SELECT distribuicao FROM lotes WHERE id = 'L0'").fetchone()[0] == "sa"
    db.conexao.close()
