from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import secrets
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError
from comum.protocolo import ConsultaExcesso, DecisaoExcesso, Heartbeat, Job, NovoLote, Progresso, Registro, Relatorio, RespostaRegistro
from comum.util import escrever_json_atomico, sha256_arquivo, validar_id
import database
import planner
from config import ConfigMestre

log = logging.getLogger("renderopt.master")
VERIFICACAO_S = 0.25
PING_S = 15

def criar_app (config: ConfigMestre) -> FastAPI:
    db = database.DB(config.caminho_db)
    config.pasta_cenas.mkdir(parents=True, exist_ok=True)
    config.pasta_resultados.mkdir(parents=True, exist_ok=True)
    parar = threading.Event()

    def checar_id (valor: str) -> str:
        try: return validar_id(valor)
        except ValueError as erro: raise HTTPException(422, str(erro)) from None

    def maquina_de(linha: sqlite3.Row) -> planner.Maquina:
        return planner.Maquina(
            id=linha["id"],
            hardware=json.loads(linha["hardware"]),
            versoes_blender=json.loads(linha["versoes_blender"]),
            dispositivo=linha["dispositivo"],
        )

    def cena_do_lote (conexao: sqlite3.Connection, lote_id: str) -> planner.Cena:
        lote = conexao.execute(
            "SELECT * FROM lotes WHERE id = ?",
            (lote_id,)
        ).fetchone()
        caminho = config.pasta_cenas / f"{lote['cena_id']}.blend"
        return planner.Cena(
            id=lote["cena_id"],
            caminho=str(caminho),
            versao=lote["versao_cena"]
        )

    def cargas (conexao: sqlite3.Connection) -> dict[str, float]:
        return {m: database.carga_pendente(conexao, m) for m in database.maquinas_disponiveis(conexao)}

    def recolocar (conexao: sqlite3.Connection, job_id: str, excluir: Optional[str]) -> None:
        """
        Coloca o job no fim da melhor fila, sem máquina elegível, ele fica órfão.
        """
        destino = planner.escolher_destino(database.estimativas_do_job(conexao, job_id), cargas(conexao), excluir)
        if destino is None:
            conexao.execute(
                "UPDATE jobs SET estado = 'PENDENTE', maquina = NULL, posicao = NULL, "
                "token = NULL, inicio = NULL WHERE id = ?",
                (job_id,)
            )
            database.registrar_evento(conexao, time.time(), "orfao", job_id)
        else:
            database.colocar_no_fim(conexao, job_id, destino)

    def verificar_fim_do_lote (conexao: sqlite3.Connection, lote_id: str) -> None:
        (restantes,) = conexao.execute(
            "SELECT COUNT(*) FROM jobs WHERE lote_id = ? AND estado IN ('PENDENTE', 'EXECUTANDO')",
            (lote_id,)
        ).fetchone()
        if restantes == 0:
            conexao.execute(
                "UPDATE lotes SET fim = ? WHERE id = ? AND fim IS NULL",
                (time.time(), lote_id)
            )
            database.registrar_evento(conexao, time.time(), "lote_concluido", dados=lote_id)

    def devolver (conexao: sqlite3.Connection, job_id: str, maquina: str, motivo: str, conta_tentativa: bool) -> None:
        """
        Job falhou ou a máquina caiu, portanto, volta para a fila de outra máquina.
        """
        job = conexao.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,)
        ).fetchone()
        tentativas = job["tentativas"] + (1 if conta_tentativa else 0)
        conexao.execute(
            "UPDATE jobs SET tentativas = ?, evitar = ? WHERE id = ?",
            (tentativas, maquina, job_id)
        )
        database.registrar_evento(conexao, time.time(), "devolvido", job_id, maquina, motivo)
        if tentativas >= config.max_tentativas:
            conexao.execute(
                "UPDATE jobs SET estado = 'FALHA_DEFINITIVA', token = NULL, posicao = NULL WHERE id = ?",
                (job_id,)
            )
            database.registrar_evento(conexao, time.time(), "falha_definitiva", job_id, maquina, motivo)
            verificar_fim_do_lote(conexao, job["lote_id"])
        else:
            recolocar(conexao, job_id, excluir=maquina)

    def adiar (conexao: sqlite3.Connection, job: sqlite3.Row, relatorio: Relatorio) -> None:
        """
        Job abortado por excesso de tempo. Corrige as estimativas e manda para o FIM da fila de outra máquina.
        """
        previsto = database.estimativas_do_job(conexao, job["id"]).get(job["maquina"], relatorio.tempo_previsto_s)
        progresso = Progresso(
            amostra_atual=relatorio.amostra_final,
            amostras_total=relatorio.amostras_total,
            tempo_preparacao_s=relatorio.tempo_preparacao_s,
            instante_amostra_s=relatorio.instante_ultima_amostra_s
        )
        total = planner.projetar_total(relatorio.tempo_total_s, progresso) or relatorio.tempo_total_s
        fator = max(total / previsto, 1.0)
        conexao.execute(
            "UPDATE jobs SET segundos = segundos * ? WHERE job_id = ?",
            (fator, job["id"])
        )
        conexao.execute(
            "UPDATE jobs SET adiamentos = adiamentos + 1, evitar = ? WHERE id = ?",
            (job["maquina"], job["id"])
        )
        database.registrar_evento(conexao, time.time(), "adiado", job["id"], job["maquina"], json.dumps({"fator": fator, "motivo": relatorio.motivo}))
        recolocar(conexao, job["id"], excluir=job["maquina"])

    def estimar_para_recem_chegado (maquina_id: str) -> None:
        """
        Maquina entrou com o lote em andamento: calcula a matriz de processamento dela para o que falta.
        """
        with db.transacao() as conexao:
            maquina = maquina_de(conexao.execute(
                "SELECT * FROM trabalhadores WHERE id = ?",
                (maquina_id,)
            ).fetchone())
            faltando = conexao.execute(
                "SELECT j.id, j.lote_id, j.frame FROM jobs j WHERE j.estado = 'PENDENTE' AND NOT EXISTS "
                "(SELECT 1 FROM estimativas e WHERE e.job_id = j.id AND e.maquinas = ?) ORDER BY j.lote_id",
                (maquina_id,)
            ).fetchall()
            por_lote: dict[str, list[sqlite3.Row]] = {}
            for linha in faltando:
                por_lote.setdefault(linha["lote_id"], []).append(linha)
            cenas = {lote_id: cena_do_lote(conexao, lote_id) for lote_id in por_lote}

        # A IA roda fora da trava, o mestre continua atendendo os outros
        novas: list[tuple[str, str, float]] = []
        for lote_id, linhas in por_lote.items():
            P = planner.estimar_matriz_p(cenas[lote_id], [l["frame"] for l in linhas], [maquina])
            novas += [(l["id"], maquina_id, float(P[i, 0])) for i, l in enumerate(linhas) if np.isfinite(P[i, 0])]

        if novas:
            with db.transacao() as conexao:
                conexao.executemany(
                    "INSERT OR IGNORE INTO estimativas (job_id, maquina, segundos) VALUES (?, ?, ?)",
                    novas
                )

    def vigiar () -> None:
        """
        A cada 5 segundos, declara caídos os trabalhadores em silêncio.
        """
        while not parar.wait(5):
            try:
                with db.transacao() as conexao:
                    agora = time.time()
                    caidos = [m for (m,) in conexao.execute(
                        "SELECT id FROM trabalhadores WHERE estado = 'DISPONIVEL' AND ultimo_heartbeat < ?",
                        (agora - config.limite_silencio_s,)
                    )]
                    # Primeiro, marca todos, para nenhum receber jobs dos outros
                    for m in caidos:
                        conexao.execute(
                            "UPDATE trabalhadores SET estado = 'INDISPONIVEL' WHERE id = ?",
                            (m,)
                        )
                        database.registrar_evento(conexao, agora, "trabalhador_caiu", maquina=m)
                        log.warning(f"Trabalhador {m} em silêncio, redistribuindo jobs dele.")

                    for m in caidos:
                        for (job_id,) in conexao.execute(
                                "SELECT id FROM jobs WHERE estado = 'EXECUTANDO' AND maquina = ?", (m,)).fetchall():
                            devolver(conexao, job_id, m, "Trabalhador caiu", conta_tentativa=True)

                        for job in database.fila(conexao, m):
                            devolver(conexao, job["id"], m, "fila redistribuída", conta_tentativa=False)
            except Exception:
                log.exception("Erro no vigia de heartbeats.");

    @asynccontextmanager
    async def ciclo_de_vida (_app: FastAPI):
        with db.transacao() as conexao:
            conexao.execute(
                "UPDATE trabalhadores SET ultimo_heartbeat = ? WHERE estado = 'DISPONIVEL'",
                (time.time(),)
            )
        threading.Thread(target=vigiar, name="vigia", daemon=True).start()
        yield
        parar.set()

    def verificar_token (x_token: Optional[str] = Header(deafult=None)) -> None:
        if config.token and not secrets.compare_digest((x_token or "").encode(), config.token.encode()):
            raise HTTPException(401, "Token inválido")

    app = FastAPI(
        title="render-opt: mestre",
        lifespan=ciclo_de_vida,
        dependencies=[Depends(verificar_token)]
    )

    @app.post("/trabalhadores/registrar", response_model=RespostaRegistro)
    def registrar (registro: Registro) -> RespostaRegistro:
        agora = time.time()
        with db.transacao() as conexao:
            conexao.execute(
                """
                INSERT INTO trabalhadores (id, hardware, versoes_blender, dispositivo, estado, ultimo_heartbeat)
                VALUES (?, ?, ?, ?, 'DISPONIVEL', ?) ON CONFLICT(id) DO
                UPDATE SET
                    hardware = excluded.hardware,
                    versoes_blender = excluded.versoes_blender,
                    dispositivo = excluded.dispositivo,
                    estado = 'DISPONIVEL',
                    ultimo_heartbeat = excluded.ultimo_heartbeat
                """,
                (
                    registro.trabalhador_id,
                    registro.hardware.model_dump_json(),
                    json.dumps(registro.versoes_blender),
                    registro.dispositivo,
                    agora
                )
            )

            # Se ele tinha um job executando, o processo anterior morreu no meio
            for (job_id,) in conexao.execute("SELECT id FROM jobs WHERE estado = 'EXECUTANDO' AND maquina = ?", (registro.trabalhador_id,)).fetchall():
                devolver(conexao, job_id, registro.trabalhador_id, "trabalhador reiniciou", conta_tentativa=True)

            database.registrar_evento(conexao, agora, "registro", maquina=registro.trabalhador_id, dados=registro.model_dump_json())

        estimar_para_recem_chegado(registro.trabalhador_id)
        log.info(f"trabalhador {registro.trabalhador_id} registrado ({registro.dispositivo}, Blender {', '.join(registro.versoes_blender)})")
        return RespostaRegistro(
            intervalo_heartbeat_s=config.intervalo_heartbeat_s,
            intervalo_ocioso_s=config.intervalo_ocioso_s,
            espera_longa_s=config.espera_longa_s
        )

    @app.post("/trabalhadores/{trabalhador_id}/heartbeat")
    def heartbeat (trabalhador_id: str, hb: Heartbeat) -> dict[str, bool]:
        with db.transacao() as conexao:
            cursor = conexao.execute(
                "UPDATE trabalhadores SET ultimo_heartbeat = ? WHERE id = ? AND estado = 'DISPONIVEL'",
                (time.time(), trabalhador_id)
            )
            if cursor.rowcount == 0:
                # Desconhecido ou caiu, é necessário se registrar novamente
                raise HTTPException(404, "Registre-se novamente.")

        return {"ok": True}

    def entregar (trabalhador_id: str) -> Optional[Job]:
        """
        Tenta entregar um job agora: própria fila, órfãos ou roubo. None = nada para ele.
        """
        agora = time.time()
        with db.transacao() as conexao:
            linha = conexao.execute(
                "SELECT estado FROM trabalhadores WHERE id = ?",
                (trabalhador_id,)
            ).fetchone()
            if linha is None or linha["estado"] != "DISPONIVEL":
                raise HTTPException(404, "Registre-se novamente.")
            conexao.execute(
                "UPDATE trabalhadores SET ultimo_heartbeat = ? WHERE id = ?",
                (agora, trabalhador_id)
            )

            # Se ainda consta um job executando, ele o perdeu (só perde o job quem está livre)
            for (job_id,) in conexao.execute(
                "SELECT id FROM jobs WHERE estado = 'EXECUTANDO' AND maquina = ?",
                (trabalhador_id,)
            ).fetchall():
                devolver(conexao, job_id, trabalhador_id, "Trabalhador pediu job sem entregar o anterior", True)

            # 1. A própria fila
            job = conexao.execute(
                "SELECT * FROM jobs WHERE estado = 'PENDENTE' AND maquina = ? ORDER BY posicao LIMIT 1",
                (trabalhador_id,)
            ).fetchone()

            # 2. Jobs órfãos para os quais ele é elegível
            if job is None:
                job = conexao.execute(
                    "SELECT j.* FROM jobs j JOIN estimativas e ON e.job_id = j.id AND e.maquina = ? "
                    "WHERE j.estado = 'PENDENTE' AND j.maquina IS NULL ORDER BY j.id LIMIT 1",
                    (trabalhador_id,)
                ).fetchone()

            # 3. Roubar do fim da fila de outra máquina
            if job is None:
                filas = {m: [j["id"] for j in database.fila(conexao, m)] for m in database.maquinas_disponiveis(conexao)}
                estimativas = {}
                for ids in filas.values():
                    if ids:
                        est = database.estimativas_do_job(conexao, ids[-1])
                        (evitar,) = conexao.execute(
                            "SELECT evitar FROM jobs WHERE id = ?",
                            (ids[-1],)
                        ).fetchone()
                        if evitar == trabalhador_id:
                            est.pop(trabalhador_id, None) # Não pega de volta o que acabou de largar
                        estimativas[ids[-1]] = est

                roubado = planner.escolher_roubo(trabalhador_id, filas, estimativas, cargas(conexao))
                if roubado is not None:
                    job = conexao.execute(
                        "SELECT * FROM jobs WHERE id = ?",
                        (roubado,)
                    ).fetchone()
                    database.registrar_evento(conexao, agora, "roubo", roubado, trabalhador_id, dados=f"de {job['maquina']}")

            if job is None:
                return None

            previsto = database.estimativas_do_job(conexao, job["id"])[trabalhador_id]
            if job["adiamentos"] >= config.max_adiamentos:
                checkpoint = None
            else:
                checkpoint = max(config.fator_excesso * previsto, previsto + config.folga_minima_s)

            teto = max(config.fator_teto * previsto, (checkpoint or previsto) + config.folga_minima_s)
            token = uuid.uuid4().hex
            conexao.execute(
                "UPDATE jobs SET estado = 'EXECUTANDO', maquina = ?, posicao = NULL, token = ?, inicio = ? WHERE id = ?",
                (trabalhador_id, token, agora, job["id"])
            )
            conexao.execute(
                "UPDATE lotes SET inicio = COALESCE(inicio, ?) WHERE id = ?",
                (agora, job["lote_id"])
            )
            lote = conexao.execute(
                "SELECT * FROM lotes WHERE id = ?",
                (job["id"],)
            ).fetchone()
            database.registrar_evento(conexao, agora, "entregue", job["id"], trabalhador_id)

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

    async def esperar_evento (visto: int, prazo: float, request: Request) -> bool:
        """
        Dorme até surgir um evento depois de `visto` (True) ou o prazo vencer (False)
        """
        while True:
            if db.ultimo_evento != visto:
                return not await request.is_disconnected()
            if time.monotonic() >= prazo or await request.is_disconnected():
                return False
            await asyncio.sleep(VERIFICACAO_S)

    @app.post("/trabalhadores/{trabalhador_id}/proximo-job")
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
            job = await run_in_threadpool(entregar, trabalhador_id)
            if job is not None:
                return job
            if not await esperar_evento(visto, prazo, request):
                return Response(status_code=204)

    @app.get("/trabalhadores")
    def listar_trabalhadores () -> list[dict]:
        with db.transacao() as conexao:
            linhas = conexao.execute(
                "SELECT id, dispositivo, estado, ultimo_heartbeat FROM trabalhadores"
            ).fetchall()
            return [dict(l) for l in linhas]

    @app.get("/cenas/{cena_id}")
    def baixar_cena(cena_id: str) -> FileResponse:
        caminho = config.pasta_cenas / f"{checar_id(cena_id)}.blend"
        if not caminho.is_file():
            raise HTTPException(404, "Cena desconhecida")
        return FileResponse(caminho, media_type="application/octet-stream", filename=caminho.name)

    @app.post("/jobs/{job_id}/excesso", response_model=DecisaoExcesso)
    def excesso (job_id: str, consulta: ConsultaExcesso) -> DecisaoExcesso:
        with db.transacao() as conexao:
            job = conexao.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,)
            ).fetchone()
            if job is None or job["estado"] != "EXECUTANDO" or job["token"] != consulta.token:
                raise HTTPException(409, "Entrega obsoleta")

            estimativas = database.estimativas_do_job(conexao, job_id)
            vivas = set(database.maquinas_disponiveis(conexao))
            outras = {m: s for m, s in estimativas.items() if m != job["maquina"] and m in vivas}
            decisao = planner.decidir_excesso(
                previsto_aqui=estimativas[job["maquina"]],
                decorrido=consulta.tempo_decorrido_s,
                progresso=consulta.progresso,
                previstos_outras=outras,
                config=config
            )
            database.registrar_evento(conexao, time.time(), "excesso", job_id, job["maquina"], decisao.model_dump_json())

        log.info(f"Job {job_id} passou do checkpoint em {job['maquina']}: {decisao.acao} ({decisao.motivo})")
        return decisao

    @app.post("/jobs/{job_id}/resultado")
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
        if rel.status == "concluido":
            if imagem is None:
                raise HTTPException(422, "job concluído sem imagem")
            temporario, hash_recebido = receber_arquivo(imagem, config.pasta_resultados)
            if hash_recebido != rel.imagem_sha256:
                temporario.unlink(missing_ok=True)
                raise HTTPException(422, "Imagem corrompida no caminho (SHA-256 não confere)")

        try:
            with db.transacao() as conexao:
                job = conexao.execute(
                    "SELECT * FROM jobs WHERE id = ?",
                    (job_id,)
                ).fetchone()
                if job is None or job["estado"] != "EXECUTANDO" or job["token"] != rel.token or job["maquina"] != rel.trabalhador_id:
                    # Resultado tardio de um job que já foi entregue a outro
                    raise HTTPException(409, "Entrega obsoleta: o job foi reatribuído")

                pasta_lote = config.pasta_resultados / job["lote_id"]
                # O JSON de toda tentativa fica guardado, inclusive abortos e falhas
                escrever_json_atomico(pasta_lote / "relatorios" / f"{job_id}_{rel.token[:8]}.json", rel.model_dump())

                if rel.status == "concluido":
                    extensao = Path(rel.image_nome or "").suffix.lower()
                    if not re.fullmatch(r"\.[a-z0-9]{1,5}", extensao):
                        extensao = ".png"
                    destino = pasta_lote / "frames" / f"frame_{job['frame']:06d}{extensao}"
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporario, destino)
                    temporario = None
                    conexao.execute(
                        "UPDATE jobs SET estado = 'CONCLUIDO', token = NULL, tempo_real_s = ? WHERE id = ?",
                        (rel.tempo_total_s, job_id)
                    )
                    database.registrar_evento(conexao, time.time(), "concluido", job_id, rel.trabalhador_id, json.dumps({
                        "real": rel.tempo_total_s,
                        "previsto": rel.tempo_previsto_s
                    }))
                    verificar_fim_do_lote(conexao, job["lote_id"])
                elif rel.status == "abortado":
                    adiar(conexao, job, rel)
                else:
                    devolver(conexao, job_id, rel.trabalhador_id, rel.motivo or "falha", conta_tentativa=True)
        finally:
            if temporario is not None:
                temporario.unlink(missing_ok=True)

        return {"ok": True}

    @app.get("/eventos")
    async def fluxo_de_eventos(
        request: Request,
        desde: Optional[int] = None,
        last_event_id: Optional[str] = Header(default=None)
    ) -> StreamingResponse:
        """
        Publica cada linha nova da tabela `eventos` no formato Server-Sent Events
        (text/event-stream). Qualquer programa pode assinar. Ao reconectar, o cliente
        manda o cabeçalho `Last-Event-ID` com o último `id` que recebeu e continua dali,
        sem perder nem repetir
        :param desde: se igual a 0, reenvia o histórico inteiro, se não for passado, somente o que vier.
        """
        if last_event_id is not None and last_event_id.isdigit():
            inicio = int(last_event_id)
        elif desde is not None:
            inicio = max(desde, 0)
        else:
            inicio = db.ultimo_evento

        def ler (depois_de: int) -> list[sqlite3.Row]:
            with db.transacao() as conexao:
                return database.eventos_depois(conexao, depois_de)

        async def fluxo ():
            ultimo, ultimo_envio = inicio, time.monotonic()
            yield "retry: 3000\n\n" # Se cair, o navegador reconecta em 3 s
            while not await request.is_disconnected():
                if db.ultimo_evento > ultimo:
                    for linha in await run_in_threadpool(ler, ultimo):
                        ultimo = linha["id"]
                        dados = json.dumps(dict(linha), ensure_ascii=False)
                        yield f"id: {linha['id']}\nevent: {linha['tipo']}\ndata: {dados}\n\n"
                        ultimo_envio = time.monotonic()
                        continue
                if time.monotonic() - ultimo_envio > PING_S:
                    yield ": ping\n\n"
                    ultimo_envio = time.monotonic()
                await asyncio.sleep(VERIFICACAO_S)

        return StreamingResponse(fluxo(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/lotes")
    def criar_lote (novo: NovoLote) -> dict:
        caminho = config.pasta_cenas / f"{novo.cena_id}.blend"
        if not caminho.is_file():
            raise HTTPException(404, f"cena {novo.cena_id} nao está em {config.pasta_cenas}")
        if sha256_arquivo(caminho) != novo.cena_sha256:
            raise HTTPException(422, "O SHA-256 da cena no mestre difere do informado")

        frames = sorted(set(novo.frames))
        with db.transacao() as conexao:
            if conexao.execute("SELECT 1 FROM lotes WHERE id = ?", (novo.lote_id,)).fetchone():
                raise HTTPException(409, "Lote já existe")
            maquinas = [maquina_de(l) for l in conexao.execute(
                "SELECT * FROM trabalhadores WHERE estado = 'DISPONIVEL' ORDER BY id"
            )]

        if not maquinas:
            raise HTTPException(409, "Nenhum trabalhador disponível.")

        cena = planner.Cena(id=novo.cena_id, caminho=str(caminho), versao=novo.versao_cena)
        P = planner.estimar_matriz_p(cena, frames, maquinas)

        try:
            plano = planner.gerar_plano_renderizacao(P)
        except ValueError as erro:
            raise HTTPException(422, str(erro)) from None

        if sorted(i for seq in plano for i in seq) != list(range(len(frames))):
            raise HTTPException(500, "O plano não cobre cada frame exatamente uma vez")

        agora = time.time()
        with db.transacao() as conexao:
            conexao.execute(
                "INSERT INTO lotes (id, cena_id, cena_sha256, versao_cena, criado_em) VALUES (?, ?, ?, ?, ?)",
                (novo.lote_id, novo.cena_id, novo.cena_sha256, novo.versao_cena, agora)
            )

            for k, sequencia in enumerate(plano):
                for posicao, i in enumerate(sequencia, start=1):
                    job_id = f"{novo.lote_id}-f{frames[i]:06d}"
                    conexao.execute(
                        "INSERT INTO jobs (id, lote_id, frame, estado, maquina, posicao) VALUES (?, ?, ?, 'PENDENTE', ?, ?)",
                        (job_id, novo.lote_id, frames[i], maquinas[k].id, posicao)
                    )
                    conexao.executemany(
                        "INSERT INTO estimativas (job_id, maquina, segundos) VALUES (?, ?, ?)",
                        [(job_id, maquinas[c].id, float(P[i, c])) for c in range(len(maquinas)) if np.isfinite(P[i, c])]
                    )

            database.registrar_evento(conexao, agora, "lote_criado", dados=novo.model_dump_json())

        tempos = [float(sum(P[i, k] for i in seq)) for k, seq in enumerate(plano)]
        return {
            "lote_id": novo.lote_id,
            "jobs": len(frames),
            "makespan_estimado_s": max(tempos),
            "filas": {maquinas[k].id: len(seq) for k, seq in enumerate(plano)},
        }

    @app.get("lotes/{lote_id")
    def status_do_lote (lote_id: str) -> dict:
        with db.transacao() as conexao:
            lote = conexao.execute(
                "SELECT * FROM lotes WHERE id = ?",
                (lote_id,)
            ).fetchone()
            if lote is None:
                raise HTTPException(404, "Lote desconhecido")
            por_estado = dict(conexao.execute(
                "SELECT estado, COUNT(*) FROM jobs WHERE lote_id = ? GROUP BY estado",
                (lote_id,)
            ).fetchall())

            por_maquina = {}
            for m, n, soma in conexao.execute(
                "SELECT maquina, COUNT(*), COALESCE(SUM(tempo_real_s), 0) FROM jobs "
                "WHERE lote_id = ? AND estado = 'CONCLUIDO' GROUP BY maquina",
                (lote_id,)
            ):
                por_maquina[m] = {"concluidos": n, "segundos_renderizando": soma}

            for m, n in conexao.execute(
                "SELECT maquina, COUNT(*) FROM jobs WHERE lote_id = ? AND estado = 'PENDENTE' GROUP BY maquina",
                    (lote_id,)
            ):
                por_maquina.setdefault(m or "(sem dono)", {})["na fila"] = n

            for m, j in conexao.execute(
                "SELECT maquina, id FROM jobs WHERE lote_id = ? AND estado = 'EXECUTANDO'",
                (lote_id,)
            ):
                por_maquina.setdefault(m, {})["executando"] = j

        inicio, fim = lote["inicio"], lote["fim"]
        return {
            "lote_id": lote_id,
            "jobs_por_estado": por_estado,
            "por_maquina": por_maquina,
            "concluido": fim is not None,
            "makespan_s": None if inicio is None else (fim or time.time()) - inicio
        }

    return app

def receber_arquivo (upload: UploadFile, pasta: Path) -> tuple[Path, str]:
    """
    Copia o upload para um temporário em `pasta`, calculando o
    SHA-256 no caminho.
    """
    fd, nome = tempfile.mkstemp(dir=pasta, suffix=".parcial")
    h = sha256()
    try:
        with os.fdopen(fd, "wb") as destino:
            while parte := upload.file.read(1 << 20):
                destino.write(parte)
                h.update(parte)
    except BaseException:
        Path(nome).unlink(missing_ok=True)
        raise
    return Path(nome), h.hexdigest()