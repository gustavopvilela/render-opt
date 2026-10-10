from __future__ import annotations
from enum import Enum

class EstadoTrabalhador (str, Enum):
    DISPONIVEL = "DISPONIVEL"
    INDISPONIVEL = "INDISPONIVEL"

class EstadoJob (str, Enum):
    PENDENTE = "PENDENTE"
    EXECUTANDO = "EXECUTANDO"
    CONCLUIDO = "CONCLUIDO"
    FALHA_DEFINITIVA = "FALHA_DEFINITIVA"

class TipoEvento (str, Enum):
    REGISTRO = "registro"
    TRABALHADOR_CAIU = "trabalhador_caiu"
    LOTE_CRIADO = "lote_criado"
    LOTE_CONCLUIDO = "lote_concluido"
    ENTREGUE = "entregue"
    ROUBO = "roubo"
    EXCESSO = "excesso"
    ADIADO = "adiado"
    DEVOLVIDO = "devolvido"
    ORFAO = "orfao"
    CONCLUIDO = "concluido"
    FALHA_DEFINITIVA = "falha_definitiva"
