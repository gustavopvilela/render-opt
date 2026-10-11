import gzip
import hashlib
import json
import math
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx2
import numpy as np
import pytest
import uvicorn
import zstandard
from fastapi.testclient import TestClient

from comum.protocolo import DecisaoExcesso, Progresso
from comum.util import sha256_arquivo, validar_id
from master import planner, submit
from master.app import criar_app
from master.config import ConfigMestre
from master.observe import assinar
from worker import blender
from worker.log_blender import AnaliseLog, segundos

RAIZ = Path(__file__).resolve().parents[1]
FALSO = [sys.executable, str(RAIZ / "tests" / "blender_falso.py")]


@pytest.mark.parametrize("valor", ["lab-pc01", "L001-f000001", "cena_1.v2"])
def test_id_valido(valor):
    assert validar_id(valor) == valor


@pytest.mark.parametrize("valor", ["", "../etc", "a/b", "a\\b", "CON", "nul.txt", "fim.", "x" * 65])
def test_id_invalido(valor):
    with pytest.raises(ValueError):
        validar_id(valor)


def test_segundos():
    assert segundos("00:01.94") == pytest.approx(1.94)
    assert segundos("1:00:00.00") == 3600


def test_analise_log_com_linhas_reais_do_blender_5_1():
    a = AnaliseLog()
    linhas = [
        (0.1, "Blender 5.1.0 (hash adfe2921d5f3 built 2026-03-17 01:37:32)"),
        (1.6, "00:03.563  render           | Fra: 1 | Mem: 284M | Sample 0/8"),
        (1.7, "00:03.672  render           | Fra: 1 | Remaining: 00:00.70 | Mem: 288M | Sample 1/8"),
        (2.2, "00:04.219  render           | Fra: 1 | Mem: 288M | Sample 8/8"),
        (2.4, "00:04.344  render           | Saved: 'C:\\tmp\\out_0001.png'"),
        (2.4, "00:04.344  render           | Time: 00:01.94 (Saving: 00:00.02)"),
    ]
    for t, linha in linhas:
        a.alimentar(linha, t)
    assert a.versao_blender == "5.1"
    assert a.tempo_preparacao_s == 1.6
    assert (a.amostra_atual, a.amostras_total) == (8, 8)
    assert a.tempo_amostragem_s == pytest.approx(0.8)
    assert a.memoria_max_mb == 288
    assert (a.tempo_blender_s, a.tempo_gravacao_s) == (pytest.approx(1.94), pytest.approx(0.02))


def test_analise_log_formato_4x():
    a = AnaliseLog()
    a.alimentar("Fra:1 Mem:24.08M (Peak 25.43M) | Time:00:00.45 | Mem:0.00M, Peak:0.00M "
                "| Scene, ViewLayer | Sample 3/128", 5.0)
    assert (a.amostra_atual, a.amostras_total, a.memoria_max_mb) == (3, 128, 24.08)
    assert a.tempo_blender_s is None


def cfg_teste(tmp_path, **extras) -> ConfigMestre:
    return ConfigMestre(pasta_dados=tmp_path / "mestre", **extras)


def test_plano_provisorio_cobre_todos_e_respeita_elegibilidade():
    P = np.array([[10, math.inf], [5, 1], [8, 2], [3, math.inf]], dtype=float)
    plano = planner.gerar_plano_renderizacao(P)
    assert sorted(i for seq in plano for i in seq) == [0, 1, 2, 3]
    assert 0 in plano[0] and 3 in plano[0]


def test_plano_recusa_frame_sem_maquina():
    with pytest.raises(ValueError):
        planner.gerar_plano_renderizacao(np.array([[math.inf, math.inf]]))


def test_elegibilidade_por_versao():
    cena = planner.Cena("c", "c.blend", "5.1")
    maquinas = [planner.Maquina("a", {}, ["4.5"], "CPU"), planner.Maquina("b", {}, ["4.5", "5.1"], "CPU")]
    assert planner.matriz_elegibilidade(cena, maquinas).tolist() == [False, True]


def test_excesso_politica_simples_sempre_aborta(tmp_path):
    d = planner.decidir_excesso(previsto_aqui=100, decorrido=160, progresso=Progresso(),
                                previstos_outras={}, config=cfg_teste(tmp_path, politica_excesso="simples"))
    assert d.acao == "abortar"


def test_excesso_quase_pronto_continua(tmp_path):
    d = planner.decidir_excesso(previsto_aqui=100, decorrido=160,
                                progresso=Progresso(amostra_atual=95, amostras_total=100, tempo_preparacao_s=10),
                                previstos_outras={"b": 100}, config=cfg_teste(tmp_path))
    assert d.acao == "continuar" and d.proximo_checkpoint_s > 160


def test_excesso_muito_lento_aborta_se_outra_e_bem_melhor(tmp_path):
    d = planner.decidir_excesso(previsto_aqui=100, decorrido=160,
                                progresso=Progresso(amostra_atual=10, amostras_total=100, tempo_preparacao_s=10),
                                previstos_outras={"b": 30}, config=cfg_teste(tmp_path))
    assert d.acao == "abortar"


def test_excesso_sem_outra_maquina_continua(tmp_path):
    d = planner.decidir_excesso(previsto_aqui=100, decorrido=160,
                                progresso=Progresso(amostra_atual=10, amostras_total=100, tempo_preparacao_s=10),
                                previstos_outras={}, config=cfg_teste(tmp_path))
    assert d.acao == "continuar"


def test_roubo_pega_o_job_que_mais_adianta():
    filas = {"a": [], "b": ["j1", "j2"], "c": ["j3"]}
    estimativas = {"j2": {"a": 10, "b": 10}, "j3": {"a": 50, "c": 5}}
    cargas = {"a": 0, "b": 100, "c": 5}
    assert planner.escolher_roubo("a", filas, estimativas, cargas) == "j2"


def test_roubo_nao_rouba_se_nao_compensa():
    filas = {"a": [], "b": ["j1"]}
    assert planner.escolher_roubo("a", filas, {"j1": {"a": 50, "b": 5}}, {"a": 0, "b": 5}) is None


@pytest.mark.parametrize("cabecalho, versao", [
    (b"BLENDER-v279REND", "2.79"),
    (b"BLENDER_v273REND", "2.73"),
    (b"BLENDER-v300REND", "3.0"),
    (b"BLENDER-v405REND", "4.5"),
    (b"BLENDER17-01v0501REND", "5.1"),
])
@pytest.mark.parametrize("compressao", ["nenhuma", "gzip", "zstd"])
def test_versao_do_arquivo_le_o_cabecalho(tmp_path, cabecalho, versao, compressao):
    dados = cabecalho + b"\0" * 64
    if compressao == "gzip":
        dados = gzip.compress(dados)
    elif compressao == "zstd":
        dados = zstandard.ZstdCompressor().compress(dados)
    arquivo = tmp_path / "cena.blend"
    arquivo.write_bytes(dados)
    assert submit.versao_do_arquivo(arquivo) == versao


def test_versao_do_arquivo_recusa_o_que_nao_e_blend(tmp_path):
    arquivo = tmp_path / "cena.blend"
    arquivo.write_bytes(b"nao sou um blend")
    with pytest.raises(SystemExit):
        submit.versao_do_arquivo(arquivo)


@pytest.mark.parametrize("versao, esperado", [("2.79", "b36"), ("3.6", "b36"), ("4.2", "b45"), ("5.1", "b51")])
def test_escolher_blender_pega_a_menor_versao_que_abre(versao, esperado):
    assert submit.escolher_blender({"5.1": "b51", "4.5": "b45", "3.6": "b36"}, versao) == esperado


def test_escolher_blender_recusa_cena_mais_nova_que_todos():
    with pytest.raises(SystemExit):
        submit.escolher_blender({"4.5": "b45", "5.1": "b51"}, "5.2")


def rodar_falso(tmp_path, monkeypatch, frame, consultar=lambda d, a: None, **limites):
    monkeypatch.setenv("FALSO_PREPARO_S", "0.1")
    monkeypatch.setenv("FALSO_AMOSTRA_S", "0.02")
    saida = tmp_path / f"saida_{frame}"
    comando = blender.montar_comando(FALSO, tmp_path / "cena.blend", saida, frame, "CPU")
    padrao = dict(checkpoint_s=None, teto_s=None, limite_sem_saida_s=30)
    return blender.executar(comando, saida, consultar=consultar, **{**padrao, **limites})


def test_execucao_normal(tmp_path, monkeypatch):
    r = rodar_falso(tmp_path, monkeypatch, 1)
    assert r.status == "concluido" and r.imagem.name == "frame_000001.png"
    assert r.analise.tempo_preparacao_s >= 0.1
    assert r.analise.amostras_total == 10


def test_execucao_falha_codigo_de_saida(tmp_path, monkeypatch):
    monkeypatch.setenv("FALSO_FALHAR", "2")
    r = rodar_falso(tmp_path, monkeypatch, 2)
    assert r.status == "falha" and r.codigo_saida == 1 and r.imagem is None


def test_execucao_travada_e_morta(tmp_path, monkeypatch):
    monkeypatch.setenv("FALSO_TRAVAR", "3")
    inicio = time.perf_counter()
    r = rodar_falso(tmp_path, monkeypatch, 3, limite_sem_saida_s=1.0)
    assert r.status == "falha" and "sem escrever" in r.motivo
    assert time.perf_counter() - inicio < 10


def test_checkpoint_abortar_mata_o_blender(tmp_path, monkeypatch):
    monkeypatch.setenv("FALSO_LENTOS", "4")
    chamadas = []

    def consultar(decorrido, analise):
        chamadas.append((decorrido, analise.amostra_atual))
        return DecisaoExcesso(acao="abortar", motivo="teste")

    r = rodar_falso(tmp_path, monkeypatch, 4, checkpoint_s=0.5, consultar=consultar)
    assert r.status == "abortado" and len(chamadas) == 1
    assert r.tempo_total_s < 3


def test_checkpoint_sem_resposta_do_mestre_nao_aborta(tmp_path, monkeypatch):
    r = rodar_falso(tmp_path, monkeypatch, 5, checkpoint_s=0.15, consultar=lambda d, a: None)
    assert r.status == "concluido"


def test_teto_mata_sem_perguntar(tmp_path, monkeypatch):
    monkeypatch.setenv("FALSO_LENTOS", "6")
    r = rodar_falso(tmp_path, monkeypatch, 6, teto_s=0.5)
    assert r.status == "falha" and "teto" in r.motivo


def registrar(cliente, tid, versoes=("5.1",), fator=1.0):
    r = cliente.post("/trabalhadores/registrar", json={
        "trabalhador_id": tid, "versoes_blender": list(versoes), "dispositivo": "CPU",
        "hardware": {"sistema": "Linux", "nome_maquina": tid, "cpu": "x", "threads": 4, "fator_velocidade": fator},
    })
    assert r.status_code == 200, r.text


def criar_lote(cliente, cfg, lote="L1", frames=range(1, 7)):
    cena = cfg.pasta_cenas / "cena.blend"
    cena.parent.mkdir(parents=True, exist_ok=True)
    cena.write_bytes(b"BLENDER-v501 conteudo falso")
    r = cliente.post("/lotes", json={"lote_id": lote, "cenas": [{
        "cena_id": "cena", "cena_sha256": sha256_arquivo(cena), "versao_cena": "5.1", "frames": list(frames)}]})
    assert r.status_code == 200, r.text
    return r.json()


def concluir(cliente, job, tid, status="concluido", tempo=1.0, png=b"\x89PNG falso"):
    rel = {"job_id": job["job_id"], "token": job["token"], "trabalhador_id": tid, "status": status,
           "cena_id": job["cena_id"], "frame": job["frame"], "dispositivo": "CPU", "sistema": "Linux",
           "tempo_previsto_s": job["tempo_previsto_s"], "tempo_total_s": tempo,
           "inicio_utc": "x", "fim_utc": "y"}
    files = None
    if status == "concluido":
        rel.update(imagem_nome="frame_000001.png", imagem_sha256=hashlib.sha256(png).hexdigest())
        files = {"imagem": ("frame_000001.png", png, "image/png")}
    return cliente.post(f"/jobs/{job['job_id']}/resultado", data={"relatorio": json.dumps(rel)}, files=files)


def test_token_obrigatorio(tmp_path):
    cfg = cfg_teste(tmp_path, token="segredo")
    with TestClient(criar_app(cfg)) as c:
        assert c.get("/trabalhadores").status_code == 401
        assert c.get("/trabalhadores", headers={"X-Token": "segredo"}).status_code == 200


def test_fluxo_completo_e_maquina_rapida_recebe_mais(tmp_path):
    cfg = cfg_teste(tmp_path)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "rapida", fator=3.0)
        registrar(c, "lenta", fator=1.0)
        resumo = criar_lote(c, cfg, frames=range(1, 9))
        assert resumo["filas"]["rapida"] > resumo["filas"]["lenta"]

        for _ in range(8):
            for tid in ("rapida", "lenta"):
                r = c.post(f"/trabalhadores/{tid}/proximo-job")
                if r.status_code == 200:
                    assert concluir(c, r.json(), tid).status_code == 200
        status = c.get("/lotes/L1").json()
        assert status["jobs_por_estado"] == {"CONCLUIDO": 8} and status["concluido"]
        assert len(list((cfg.pasta_resultados / "L1" / "frames" / "cena").iterdir())) == 8
        assert len(list((cfg.pasta_resultados / "L1" / "relatorios").iterdir())) == 8


def test_resultado_tardio_e_descartado(tmp_path):
    cfg = cfg_teste(tmp_path)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "a")
        criar_lote(c, cfg, frames=[1])
        job = c.post("/trabalhadores/a/proximo-job").json()
        registrar(c, "a")
        assert concluir(c, job, "a").status_code == 409


def test_imagem_corrompida_e_recusada(tmp_path):
    cfg = cfg_teste(tmp_path)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "a")
        criar_lote(c, cfg, frames=[1])
        job = c.post("/trabalhadores/a/proximo-job").json()
        rel = {"job_id": job["job_id"], "token": job["token"], "trabalhador_id": "a", "status": "concluido",
               "cena_id": "cena", "frame": 1, "dispositivo": "CPU", "sistema": "Linux",
               "tempo_previsto_s": 1, "tempo_total_s": 1, "inicio_utc": "x", "fim_utc": "y",
               "imagem_nome": "f.png", "imagem_sha256": hashlib.sha256(b"original").hexdigest()}
        r = c.post(f"/jobs/{job['job_id']}/resultado", data={"relatorio": json.dumps(rel)},
                   files={"imagem": ("f.png", b"corrompido", "image/png")})
        assert r.status_code == 422


def test_abortado_vai_para_o_fim_de_outra_fila_e_depois_roda_sem_checkpoint(tmp_path):
    cfg = cfg_teste(tmp_path)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "a")
        registrar(c, "b")
        criar_lote(c, cfg, frames=[1, 2, 3, 4])
        job = c.post("/trabalhadores/a/proximo-job").json()
        previsto = job["tempo_previsto_s"]
        assert job["checkpoint_s"] == pytest.approx(max(cfg.fator_excesso * previsto, previsto + cfg.folga_minima_s))
        assert concluir(c, job, "a", status="abortado", tempo=95).status_code == 200

        fila_b = []
        while (r := c.post("/trabalhadores/b/proximo-job")).status_code == 200:
            fila_b.append(r.json())
            concluir(c, r.json(), "b")
        assert fila_b[-1]["job_id"] == job["job_id"]
        assert fila_b[-1]["checkpoint_s"] is None
        assert fila_b[-1]["tempo_previsto_s"] > previsto


def test_falhas_repetidas_viram_falha_definitiva(tmp_path):
    cfg = cfg_teste(tmp_path, max_tentativas=2)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "a")
        criar_lote(c, cfg, frames=[1])
        for _ in range(2):
            job = c.post("/trabalhadores/a/proximo-job").json()
            concluir(c, job, "a", status="falha")
        status = c.get("/lotes/L1").json()
        assert status["jobs_por_estado"] == {"FALHA_DEFINITIVA": 1} and status["concluido"]


def test_trabalhador_em_silencio_tem_jobs_redistribuidos(tmp_path):
    cfg = cfg_teste(tmp_path, limite_silencio_s=0.5)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "a")
        registrar(c, "b")
        criar_lote(c, cfg, frames=range(1, 5))
        c.post("/trabalhadores/a/proximo-job")
        estados = {}
        prazo = time.time() + 15
        while time.time() < prazo:
            c.post("/trabalhadores/b/heartbeat", json={})
            estados = {t["id"]: t["estado"] for t in c.get("/trabalhadores").json()}
            if estados["a"] == "INDISPONIVEL":
                break
            time.sleep(0.2)
        assert estados["a"] == "INDISPONIVEL"
        status = c.get("/lotes/L1").json()
        assert set(status["por_maquina"]) == {"b"}
        assert status["por_maquina"]["b"]["na fila"] == 4


def test_projecao_usa_o_instante_da_linha_e_nao_explode(tmp_path):
    p = Progresso(amostra_atual=1, amostras_total=100, tempo_preparacao_s=10, instante_amostra_s=12)
    assert planner.projetar_total(160, p) == pytest.approx(210)
    d = planner.decidir_excesso(previsto_aqui=100, decorrido=160, progresso=p,
                                previstos_outras={"b": 100}, config=cfg_teste(tmp_path))
    assert d.acao == "continuar"


def test_projecao_atrasada_vira_piso():
    p = Progresso(amostra_atual=50, amostras_total=100, tempo_preparacao_s=0, instante_amostra_s=10)
    assert planner.projetar_total(300, p) == 300


def test_continuar_sobe_o_teto(tmp_path, monkeypatch):
    monkeypatch.setenv("FALSO_LENTOS", "7")
    decisao = DecisaoExcesso(acao="continuar", motivo="t", proximo_checkpoint_s=100, novo_teto_s=100)
    r = rodar_falso(tmp_path, monkeypatch, 7, checkpoint_s=0.3, teto_s=1.0, consultar=lambda d, a: decisao)
    assert r.status == "concluido"


def test_quem_abortou_nao_rouba_o_job_de_volta(tmp_path):
    cfg = cfg_teste(tmp_path)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "a")
        registrar(c, "b")
        criar_lote(c, cfg, frames=[1, 2])
        job = c.post("/trabalhadores/a/proximo-job").json()
        concluir(c, job, "a", status="abortado", tempo=95)
        assert c.post("/trabalhadores/a/proximo-job").status_code == 204


@contextmanager
def mestre_de_verdade(cfg):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        porta = s.getsockname()[1]
    servidor = uvicorn.Server(uvicorn.Config(criar_app(cfg), host="127.0.0.1", port=porta,
                                             log_level="warning", timeout_graceful_shutdown=2))
    thread = threading.Thread(target=servidor.run, daemon=True)
    thread.start()
    prazo = time.time() + 10
    while not servidor.started and time.time() < prazo:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{porta}"
    finally:
        servidor.should_exit = True
        thread.join(10)


def test_espera_longa_vence_o_prazo_e_respeita_o_limite(tmp_path):
    cfg = cfg_teste(tmp_path, espera_longa_s=1.0)
    with TestClient(criar_app(cfg)) as c:
        registrar(c, "a")
        inicio = time.perf_counter()
        r = c.post("/trabalhadores/a/proximo-job", params={"espera_s": 999})
        assert r.status_code == 204
        assert 0.9 < time.perf_counter() - inicio < 5


def test_espera_longa_entrega_assim_que_o_lote_chega(tmp_path):
    cfg = cfg_teste(tmp_path)
    with mestre_de_verdade(cfg) as url, httpx2.Client(base_url=url, timeout=60) as c:
        registrar(c, "a")
        resposta = {}

        def pedir():
            inicio = time.perf_counter()
            r = c.post("/trabalhadores/a/proximo-job", params={"espera_s": 20})
            resposta.update(status=r.status_code, segundos=time.perf_counter() - inicio)

        t = threading.Thread(target=pedir)
        t.start()
        time.sleep(1.0)
        criar_lote(c, cfg)
        t.join(20)
        assert resposta["status"] == 200
        assert 0.9 < resposta["segundos"] < 3


def test_fluxo_de_eventos_e_reconexao_com_last_event_id(tmp_path):
    cfg = cfg_teste(tmp_path)
    with mestre_de_verdade(cfg) as url, httpx2.Client(base_url=url) as c:
        registrar(c, "a")
        registrar(c, "b")
        eventos = assinar(url, None, desde=0)
        primeiro, segundo = next(eventos), next(eventos)
        eventos.close()
        assert (primeiro["tipo"], primeiro["maquina"]) == ("registro", "a")
        assert (segundo["tipo"], segundo["maquina"]) == ("registro", "b")

        eventos = assinar(url, None, desde=primeiro["id"])
        assert next(eventos)["id"] == segundo["id"]
        criar_lote(c, cfg)
        assert next(eventos)["tipo"] == "lote_criado"
        eventos.close()
