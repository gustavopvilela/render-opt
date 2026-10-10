from __future__ import annotations
from fastapi import HTTPException
from comum.util import validar_id

def checar_id (valor: str) -> str:
    try: return validar_id(valor)
    except ValueError as erro: raise HTTPException(422, str(erro)) from None
