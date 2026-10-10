from __future__ import annotations
from enum import Enum

class StatusRelatorio (str, Enum):
    CONCLUIDO = "concluido"
    ABORTADO = "abortado"
    FALHA = "falha"
