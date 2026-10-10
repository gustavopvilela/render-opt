from __future__ import annotations
import json
import logging
import time
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from comum.protocolo import Heartbeat, Registro, RespostaRegistro
import master.database as database
from master.config import ConfigMestre
from master.controllers.validacao import checar_id
from master.enums import TipoEvento
from master.queries import eventos as eventos_sql
from master.queries import jobs as jobs_sql
from master.queries import trabalhadores as trabalhadores_sql
from master.services.eventos import esperar_evento
from master.services.jobs import devolver, entregar
from master.services.trabalhadores import estimar_para_recem_chegado

log = logging.getLogger("renderopt.master")

def criar_router (config: ConfigMestre, db: database.DB) -> APIRouter:
    router = APIRouter()

    @router.post("/trabalhadores/registrar", response_model=RespostaRegistro)
    def registrar (registro: Registro) -> RespostaRegistro:
        agora = time.time()
        with db.transacao() as conexao:
            trabalhadores_sql.salvar_registro(
                conexao,
                registro.trabalhador_id,
                registro.hardware.model_dump_json(),
                json.dumps(registro.versoes_blender),
                registro.dispositivo,
                agora
            )

            # Se ele tinha um job executando, o processo anterior morreu no meio
            for job_id in jobs_sql.executando_na_maquina(conexao, registro.trabalhador_id):
                devolver(config, conexao, job_id, registro.trabalhador_id, "trabalhador reiniciou", conta_tentativa=True)

            eventos_sql.registrar_evento(conexao, agora, TipoEvento.REGISTRO, maquina=registro.trabalhador_id, dados=registro.model_dump_json())

        estimar_para_recem_chegado(config, db, registro.trabalhador_id)
        log.info(f"trabalhador {registro.trabalhador_id} registrado ({registro.dispositivo}, Blender {', '.join(registro.versoes_blender)})")
        return RespostaRegistro(
            intervalo_heartbeat_s=config.intervalo_heartbeat_s,
            intervalo_ocioso_s=config.intervalo_ocioso_s,
            espera_longa_s=config.espera_longa_s
        )

    @router.post("/trabalhadores/{trabalhador_id}/heartbeat")
    def heartbeat (trabalhador_id: str, hb: Heartbeat) -> dict[str, bool]:
        with db.transacao() as conexao:
            atualizados = trabalhadores_sql.renovar_heartbeat_se_disponivel(conexao, trabalhador_id, time.time())
            if atualizados == 0:
                # Desconhecido ou caiu, é necessário se registrar novamente
                raise HTTPException(404, "Registre-se novamente.")

        return {"ok": True}

    @router.post("/trabalhadores/{trabalhador_id}/proximo-job")
    async def proximo_job (trabalhador_id: str, request: Request, espera_s: float = 0.0):
        """
        Espera longa (long polling): se não há job agora, a resposta fica pendurada até
        `espera_s` segundos (limitado a config.espera_longa_s) e sai assim que um novo
        evento trouxer trabalho (lote criado, job devolvido, maquina caída, etc.). Vencido
        o prazo sem nada, responde 204 e o trabalhador pergunta de novo.
        """
        checar_id(trabalhador_id)
        prazo = time.monotonic() + min(max(espera_s, 0.0), config.espera_longa_s)
        while True:
            visto = db.ultimo_evento()
            job = await run_in_threadpool(entregar, config, db, trabalhador_id)
            if job is not None:
                return job
            if not await esperar_evento(db, visto, prazo, request):
                return Response(status_code=204)

    @router.get("/trabalhadores")
    def listar_trabalhadores () -> list[dict]:
        with db.transacao() as conexao:
            linhas = trabalhadores_sql.listar(conexao)
            return [dict(l) for l in linhas]

    return router
