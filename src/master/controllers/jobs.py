from __future__ import annotations
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from comum.protocolo import ConsultaExcesso, DecisaoExcesso, Relatorio
from comum.util import escrever_json_atomico
import master.database as database
import master.planner as planner
from master.config import ConfigMestre
from comum.enums import StatusRelatorio
from master.enums import EstadoJob, TipoEvento
from master.queries import estimativas as estimativas_sql
from master.queries import eventos as eventos_sql
from master.queries import jobs as jobs_sql
from master.queries import trabalhadores as trabalhadores_sql
from master.services.arquivos import receber_arquivo
from master.services.jobs import adiar, devolver
from master.services.lotes import verificar_fim_do_lote

log = logging.getLogger("renderopt.master")

def criar_router (config: ConfigMestre, db: database.DB) -> APIRouter:
    router = APIRouter()

    @router.post("/jobs/{job_id}/excesso", response_model=DecisaoExcesso)
    def excesso (job_id: str, consulta: ConsultaExcesso) -> DecisaoExcesso:
        with db.transacao() as conexao:
            job = jobs_sql.buscar(conexao, job_id)
            if job is None or job["estado"] != EstadoJob.EXECUTANDO or job["token"] != consulta.token:
                raise HTTPException(409, "Entrega obsoleta")

            estimativas = estimativas_sql.estimativas_do_job(conexao, job_id)
            vivas = set(trabalhadores_sql.maquinas_disponiveis(conexao))
            outras = {m: s for m, s in estimativas.items() if m != job["maquina"] and m in vivas}
            decisao = planner.decidir_excesso(
                previsto_aqui=estimativas[job["maquina"]],
                decorrido=consulta.tempo_decorrido_s,
                progresso=consulta.progresso,
                previstos_outras=outras,
                config=config
            )
            eventos_sql.registrar_evento(conexao, time.time(), TipoEvento.EXCESSO, job_id, job["maquina"], decisao.model_dump_json())

        log.info(f"Job {job_id} passou do checkpoint em {job['maquina']}: {decisao.acao} ({decisao.motivo})")
        return decisao

    @router.post("/jobs/{job_id}/resultado")
    def resultado(
            job_id: str,
            relatorio: str = Form(...),
            imagem: Optional[UploadFile] = File(None)
    ) -> dict:
        try:
            rel = Relatorio.model_validate_json(relatorio)
        except ValidationError as erro:
            raise HTTPException(422, str(erro)) from None
        if rel.job_id != job_id:
            raise HTTPException(422, "job_id do relatório diferente da URL")

        """
        A imagem vai para um arquivo temporário antes de pegar a trava do banco,
        receber alguns MB pela rede não deve travar o resto do mestre.
        """
        temporario: Optional[Path] = None
        if rel.status == StatusRelatorio.CONCLUIDO:
            if imagem is None:
                raise HTTPException(422, "job concluído sem imagem")
            temporario, hash_recebido = receber_arquivo(imagem, config.pasta_resultados)
            if hash_recebido != rel.imagem_sha256:
                temporario.unlink(missing_ok=True)
                raise HTTPException(422, "Imagem corrompida no caminho (SHA-256 não confere)")

        try:
            with db.transacao() as conexao:
                job = jobs_sql.buscar(conexao, job_id)
                if job is None or job["estado"] != EstadoJob.EXECUTANDO or job["token"] != rel.token or job["maquina"] != rel.trabalhador_id:
                    # Resultado tardio de um job que já foi entregue a outro
                    raise HTTPException(409, "Entrega obsoleta: o job foi reatribuído")

                pasta_lote = config.pasta_resultados / job["lote_id"]
                # O JSON de toda tentativa fica guardado, inclusive abortos e falhas
                escrever_json_atomico(pasta_lote / "relatorios" / f"{job_id}_{rel.token[:8]}.json", rel.model_dump())

                if rel.status == StatusRelatorio.CONCLUIDO:
                    extensao = Path(rel.image_nome or "").suffix.lower()
                    if not re.fullmatch(r"\.[a-z0-9]{1,5}", extensao):
                        extensao = ".png"
                    destino = pasta_lote / "frames" / f"frame_{job['frame']:06d}{extensao}"
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporario, destino)
                    temporario = None
                    jobs_sql.marcar_concluido(conexao, job_id, rel.tempo_total_s)
                    eventos_sql.registrar_evento(conexao, time.time(), TipoEvento.CONCLUIDO, job_id, rel.trabalhador_id, json.dumps({
                        "real": rel.tempo_total_s,
                        "previsto": rel.tempo_previsto_s
                    }))
                    verificar_fim_do_lote(conexao, job["lote_id"])
                elif rel.status == StatusRelatorio.ABORTADO:
                    adiar(conexao, job, rel)
                else:
                    devolver(config, conexao, job_id, rel.trabalhador_id, rel.motivo or "falha", conta_tentativa=True)
        finally:
            if temporario is not None:
                temporario.unlink(missing_ok=True)

        return {"ok": True}

    return router
