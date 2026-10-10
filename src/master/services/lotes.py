from __future__ import annotations
import sqlite3
import time
import master.planner as planner
from master.config import ConfigMestre
from master.enums import TipoEvento
from master.queries import eventos as eventos_sql
from master.queries import jobs as jobs_sql
from master.queries import lotes as lotes_sql

def cena_do_lote (config: ConfigMestre, conexao: sqlite3.Connection, lote_id: str) -> planner.Cena:
    lote = lotes_sql.buscar(conexao, lote_id)
    caminho = config.pasta_cenas / f"{lote['cena_id']}.blend"
    return planner.Cena(
        id=lote["cena_id"],
        caminho=str(caminho),
        versao=lote["versao_cena"]
    )

def verificar_fim_do_lote (conexao: sqlite3.Connection, lote_id: str) -> None:
    restantes = jobs_sql.restantes_do_lote(conexao, lote_id)
    if restantes == 0:
        lotes_sql.marcar_fim(conexao, lote_id, time.time())
        eventos_sql.registrar_evento(conexao, time.time(), TipoEvento.LOTE_CONCLUIDO, dados=lote_id)
