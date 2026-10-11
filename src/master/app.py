from __future__ import annotations
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import Depends, FastAPI, Header, HTTPException
import master.database as database
from master.config import ConfigMestre
from master.controllers import cenas, eventos, jobs, lotes, trabalhadores
from master.queries import trabalhadores as trabalhadores_sql
from master.services.trabalhadores import vigiar

def criar_app (config: ConfigMestre) -> FastAPI:
    db = database.DB(config.caminho_db)
    config.pasta_cenas.mkdir(parents=True, exist_ok=True)
    config.pasta_resultados.mkdir(parents=True, exist_ok=True)
    parar = threading.Event()

    @asynccontextmanager
    async def ciclo_de_vida (_app: FastAPI):
        with db.transacao() as conexao:
            trabalhadores_sql.renovar_heartbeat_dos_disponiveis(conexao, time.time())
        threading.Thread(target=vigiar, args=(config, db, parar), name="vigia", daemon=True).start()
        yield
        parar.set()

    def verificar_token (x_token: Optional[str] = Header(default=None)) -> None:
        if config.token and not secrets.compare_digest((x_token or "").encode(), config.token.encode()):
            raise HTTPException(401, "Token inválido")

    app = FastAPI(
        title="render-opt: mestre",
        lifespan=ciclo_de_vida,
        dependencies=[Depends(verificar_token)]
    )

    app.include_router(trabalhadores.criar_router(config, db))
    app.include_router(cenas.criar_router(config))
    app.include_router(jobs.criar_router(config, db))
    app.include_router(eventos.criar_router(db))
    app.include_router(lotes.criar_router(config, db))

    return app
