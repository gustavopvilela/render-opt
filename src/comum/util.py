from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PADRAO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_RESERVADOS_WINDOWS = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefixo}{n}" for prefixo in ("COM", "LPT") for n in range(1, 10)
}

def validar_id (valor: str) -> str:
    if (
        not _PADRAO_ID.fullmatch(valor)
        or valor.endswith(".")
        or valor.split(".")[0].upper() in _RESERVADOS_WINDOWS
    ):
        raise ValueError(f"Indentificador inválido: {valor!r}")
    return valor

def sha256_arquivo (caminho: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for parte in iter(lambda: arquivo.read(bloco), b""):
            h.update(parte)
    return h.hexdigest()

def escrever_arquivo_atomico (destino: Path, dados: bytes) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    fd, temporario = tempfile.mkstemp(dir=destino.parent, prefix=destino.name, suffix=".parcial")

    try:
        with os.fdopen(fd, "wb") as arquivo:
            arquivo.write(dados)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, destino)
    except BaseException:
        Path(temporario).unlink(missing_ok=True)
        raise

def escrever_json_atomico (destino: Path, objeto: Any) -> None:
    texto = json.dumps(objeto, ensure_ascii=False, indent=2)
    escrever_arquivo_atomico(destino, texto.encode("utf-8"))

def agora_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def versao_tupla (versao: str) -> tuple[int, int]:
    maior, menor = versao.split(".")[:2]
    return int(maior), int(menor)