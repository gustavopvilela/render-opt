from __future__ import annotations
import json
import logging
import sqlite3
import threading
import time
import numpy as np
import master.database as database
import master.planner as planner
from master.config import ConfigMestre
from master.enums import EstadoTrabalhador, TipoEvento
from master.queries import estimativas as estimativas_sql
from master.queries import eventos as eventos_sql
from master.queries import jobs as jobs_sql
from master.queries import trabalhadores as trabalhadores_sql
from master.services.jobs import devolver
from master.services.lotes import cena_do_lote

log = logging.getLogger("renderopt.master")

def maquina_de(linha: sqlite3.Row) -> planner.Maquina:
    return planner.Maquina(
        id=linha["id"],
        hardware=json.loads(linha["hardware"]),
        versoes_blender=json.loads(linha["versoes_blender"]),
        dispositivo=linha["dispositivo"],
    )

def estimar_para_recem_chegado (config: ConfigMestre, db: database.DB, maquina_id: str) -> None:
    """
    Maquina entrou com o lote em andamento: calcula a matriz de processamento dela para o que falta.
    """
    with db.transacao() as conexao:
        maquina = maquina_de(trabalhadores_sql.buscar(conexao, maquina_id))
        faltando = jobs_sql.pendentes_sem_estimativa(conexao, maquina_id)
        por_lote: dict[str, list[sqlite3.Row]] = {}
        for linha in faltando:
            por_lote.setdefault(linha["lote_id"], []).append(linha)
        cenas = {lote_id: cena_do_lote(config, conexao, lote_id) for lote_id in por_lote}

    # A IA roda fora da trava, o mestre continua atendendo os outros
    novas: list[tuple[str, str, float]] = []
    for lote_id, linhas in por_lote.items():
        P = planner.estimar_matriz_p(cenas[lote_id], [l["frame"] for l in linhas], [maquina])
        novas += [(l["id"], maquina_id, float(P[i, 0])) for i, l in enumerate(linhas) if np.isfinite(P[i, 0])]

    if novas:
        with db.transacao() as conexao:
            estimativas_sql.inserir_se_ausentes(conexao, novas)

def vigiar (config: ConfigMestre, db: database.DB, parar: threading.Event) -> None:
    """
    A cada 5 segundos, declara caídos os trabalhadores em silêncio.
    """
    while not parar.wait(5):
        try:
            with db.transacao() as conexao:
                agora = time.time()
                caidos = trabalhadores_sql.silenciosos(conexao, agora - config.limite_silencio_s)
                # Primeiro, marca todos, para nenhum receber jobs dos outros
                for m in caidos:
                    trabalhadores_sql.marcar_estado(conexao, m, EstadoTrabalhador.INDISPONIVEL)
                    eventos_sql.registrar_evento(conexao, agora, TipoEvento.TRABALHADOR_CAIU, maquina=m)
                    log.warning(f"Trabalhador {m} em silêncio, redistribuindo jobs dele.")

                for m in caidos:
                    for job_id in jobs_sql.executando_na_maquina(conexao, m):
                        devolver(config, conexao, job_id, m, "Trabalhador caiu", conta_tentativa=True)

                    for job in jobs_sql.fila(conexao, m):
                        devolver(config, conexao, job["id"], m, "fila redistribuída", conta_tentativa=False)
        except Exception:
            log.exception("Erro no vigia de heartbeats.");
