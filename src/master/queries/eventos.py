from __future__ import annotations
import sqlite3
from typing import Optional
from master.enums import TipoEvento

def eventos_depois (conexao: sqlite3.Connection, id_evento: int, limite: int = 500) -> list[sqlite3.Row]:
    return conexao.execute(
        "SELECT * FROM eventos WHERE id > ? ORDER BY id LIMIT ?",
        (id_evento, limite),
    ).fetchall()

def registrar_evento(
    conexao: sqlite3.Connection,
    t: float,
    tipo: TipoEvento,
    job_id: Optional[str] = None,
    maquina: Optional[str] = None,
    dados: Optional[str] = None
) -> None:
    conexao.execute(
        "INSERT INTO eventos (t, tipo, job_id, maquina, dados) VALUES (?, ?, ?, ?, ?)",
        (t, tipo, job_id, maquina, dados),
    )
