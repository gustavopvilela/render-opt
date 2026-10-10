from __future__ import annotations
import os
import tempfile
from hashlib import sha256
from pathlib import Path
from fastapi import UploadFile

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
