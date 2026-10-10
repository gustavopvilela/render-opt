from __future__ import annotations
import json
import sqlite3
import time
import uuid
from typing import Optional
from fastapi import HTTPException
from comum.protocolo import Job, Progresso, Relatorio
import master.database as database
import master.planner as planner
from master.config import ConfigMestre
from master.enums import EstadoTrabalhador, TipoEvento
from master.queries import estimativas as estimativas_sql
from master.queries import eventos as eventos_sql
from master.queries import jobs as jobs_sql
from master.queries import lotes as lotes_sql
from master.queries import trabalhadores as trabalhadores_sql
from master.services.lotes import verificar_fim_do_lote

def cargas (conexao: sqlite3.Connection) -> dict[str, float]:
    return {m: estimativas_sql.carga_pendente(conexao, m) for m in trabalhadores_sql.maquinas_disponiveis(conexao)}

def recolocar (conexao: sqlite3.Connection, job_id: str, excluir: Optional[str]) -> None:
    """
    Coloca o job no fim da melhor fila, sem máquina elegível, ele fica órfão.
    """
    destino = planner.escolher_destino(estimativas_sql.estimativas_do_job(conexao, job_id), cargas(conexao), excluir)
    if destino is None:
        jobs_sql.tornar_orfao(conexao, job_id)
        eventos_sql.registrar_evento(conexao, time.time(), TipoEvento.ORFAO, job_id)
    else:
        jobs_sql.colocar_no_fim(conexao, job_id, destino)

def devolver (config: ConfigMestre, conexao: sqlite3.Connection, job_id: str, maquina: str, motivo: str, conta_tentativa: bool) -> None:
    """
    Job falhou ou a máquina caiu, portanto, volta para a fila de outra máquina.
    """
    job = jobs_sql.buscar(conexao, job_id)
    tentativas = job["tentativas"] + (1 if conta_tentativa else 0)
    jobs_sql.atualizar_tentativas(conexao, job_id, tentativas, maquina)
    eventos_sql.registrar_evento(conexao, time.time(), TipoEvento.DEVOLVIDO, job_id, maquina, motivo)
    if tentativas >= config.max_tentativas:
        jobs_sql.marcar_falha_definitiva(conexao, job_id)
        eventos_sql.registrar_evento(conexao, time.time(), TipoEvento.FALHA_DEFINITIVA, job_id, maquina, motivo)
        verificar_fim_do_lote(conexao, job["lote_id"])
    else:
        recolocar(conexao, job_id, excluir=maquina)

def adiar (conexao: sqlite3.Connection, job: sqlite3.Row, relatorio: Relatorio) -> None:
    """
    Job abortado por excesso de tempo. Corrige as estimativas e manda para o FIM da fila de outra máquina.
    """
    previsto = estimativas_sql.estimativas_do_job(conexao, job["id"]).get(job["maquina"], relatorio.tempo_previsto_s)
    progresso = Progresso(
        amostra_atual=relatorio.amostra_final,
        amostras_total=relatorio.amostras_total,
        tempo_preparacao_s=relatorio.tempo_preparacao_s,
        instante_amostra_s=relatorio.instante_ultima_amostra_s
    )
    total = planner.projetar_total(relatorio.tempo_total_s, progresso) or relatorio.tempo_total_s
    fator = max(total / previsto, 1.0)
    estimativas_sql.multiplicar_segundos(conexao, job["id"], fator)
    jobs_sql.registrar_adiamento(conexao, job["id"], job["maquina"])
    eventos_sql.registrar_evento(conexao, time.time(), TipoEvento.ADIADO, job["id"], job["maquina"], json.dumps({"fator": fator, "motivo": relatorio.motivo}))
    recolocar(conexao, job["id"], excluir=job["maquina"])

def entregar (config: ConfigMestre, db: database.DB, trabalhador_id: str) -> Optional[Job]:
    """
    Tenta entregar um job agora: própria fila, órfãos ou roubo. None = nada para ele.
    """
    agora = time.time()
    with db.transacao() as conexao:
        linha = trabalhadores_sql.buscar_estado(conexao, trabalhador_id)
        if linha is None or linha["estado"] != EstadoTrabalhador.DISPONIVEL:
            raise HTTPException(404, "Registre-se novamente.")
        trabalhadores_sql.renovar_heartbeat(conexao, trabalhador_id, agora)

        # Se ainda consta um job executando, ele o perdeu (só perde o job quem está livre)
        for job_id in jobs_sql.executando_na_maquina(conexao, trabalhador_id):
            devolver(config, conexao, job_id, trabalhador_id, "Trabalhador pediu job sem entregar o anterior", True)

        # 1. A própria fila
        job = jobs_sql.primeiro_da_fila(conexao, trabalhador_id)

        # 2. Jobs órfãos para os quais ele é elegível
        if job is None:
            job = jobs_sql.orfao_elegivel(conexao, trabalhador_id)

        # 3. Roubar do fim da fila de outra máquina
        if job is None:
            filas = {m: [j["id"] for j in jobs_sql.fila(conexao, m)] for m in trabalhadores_sql.maquinas_disponiveis(conexao)}
            estimativas = {}
            for ids in filas.values():
                if ids:
                    est = estimativas_sql.estimativas_do_job(conexao, ids[-1])
                    evitar = jobs_sql.evitar_do_job(conexao, ids[-1])
                    if evitar == trabalhador_id:
                        est.pop(trabalhador_id, None) # Não pega de volta o que acabou de largar
                    estimativas[ids[-1]] = est

            roubado = planner.escolher_roubo(trabalhador_id, filas, estimativas, cargas(conexao))
            if roubado is not None:
                job = jobs_sql.buscar(conexao, roubado)
                eventos_sql.registrar_evento(conexao, agora, TipoEvento.ROUBO, roubado, trabalhador_id, dados=f"de {job['maquina']}")

        if job is None:
            return None

        previsto = estimativas_sql.estimativas_do_job(conexao, job["id"])[trabalhador_id]
        if job["adiamentos"] >= config.max_adiamentos:
            checkpoint = None
        else:
            checkpoint = max(config.fator_excesso * previsto, previsto + config.folga_minima_s)

        teto = max(config.fator_teto * previsto, (checkpoint or previsto) + config.folga_minima_s)
        token = uuid.uuid4().hex
        jobs_sql.marcar_executando(conexao, job["id"], trabalhador_id, token, agora)
        lotes_sql.marcar_inicio(conexao, job["lote_id"], agora)
        lote = lotes_sql.buscar(conexao, job["id"])
        eventos_sql.registrar_evento(conexao, agora, TipoEvento.ENTREGUE, job["id"], trabalhador_id)

    return Job(
        job_id=job["id"],
        token=token,
        lote_id=lote["id"],
        cena_id=lote["cena_id"],
        cena_sha256=lote["cena_sha256"],
        versao_cena=lote["versao_cena"],
        frame=job["frame"],
        tempo_previsto_s=previsto,
        checkpoint_s=checkpoint,
        teto_s=teto,
        limite_sem_saida_s=config.limite_sem_saida_s
    )
