from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from master.config import ConfigMestre
from master.controllers.validacao import checar_id

def criar_router (config: ConfigMestre) -> APIRouter:
    router = APIRouter()

    @router.get("/cenas/{cena_id}")
    def baixar_cena(cena_id: str) -> FileResponse:
        caminho = config.pasta_cenas / f"{checar_id(cena_id)}.blend"
        if not caminho.is_file():
            raise HTTPException(404, "Cena desconhecida")
        return FileResponse(caminho, media_type="application/octet-stream", filename=caminho.name)

    return router
