from __future__ import annotations
import asyncio
import json
import sqlite3
import time
from typing import Optional
from fastapi import APIRouter, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
import master.database as database
from master.queries import eventos as eventos_sql
from master.services.eventos import PING_S, VERIFICACAO_S

def criar_router (db: database.DB) -> APIRouter:
    router = APIRouter()

    @router.get("/eventos")
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
                return eventos_sql.eventos_depois(conexao, depois_de)

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

    return router
