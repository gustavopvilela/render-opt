from __future__ import annotations
import asyncio
import time
from fastapi import Request
import master.database as database

VERIFICACAO_S = 0.25
PING_S = 15

async def esperar_evento (db: database.DB, visto: int, prazo: float, request: Request) -> bool:
    """
    Dorme até surgir um evento depois de `visto` (True) ou o prazo vencer (False)
    """
    while True:
        if db.ultimo_evento != visto:
            return not await request.is_disconnected()
        if time.monotonic() >= prazo or await request.is_disconnected():
            return False
        await asyncio.sleep(VERIFICACAO_S)
