from __future__ import annotations
import sqlite3
from typing import Optional

def buscar (conexao: sqlite3.Connection, lote_id: str) -> Optional[sqlite3.Row]:
    return conexao.execute(
        "SELECT * FROM lotes WHERE id = ?",
        (lote_id,)
    ).fetchone()

def existe (conexao: sqlite3.Connection, lote_id: str) -> bool:
    return conexao.execute("SELECT 1 FROM lotes WHERE id = ?", (lote_id,)).fetchone() is not None

def inserir (
    conexao: sqlite3.Connection,
    lote_id: str,
    cena_id: str,
    cena_sha256: str,
    versao_cena: str,
    criado_em: float
) -> None:
    conexao.execute(
        "INSERT INTO lotes (id, cena_id, cena_sha256, versao_cena, criado_em) VALUES (?, ?, ?, ?, ?)",
        (lote_id, cena_id, cena_sha256, versao_cena, criado_em)
    )

def marcar_inicio (conexao: sqlite3.Connection, lote_id: str, agora: float) -> None:
    conexao.execute(
        "UPDATE lotes SET inicio = COALESCE(inicio, ?) WHERE id = ?",
        (agora, lote_id)
    )

def marcar_fim (conexao: sqlite3.Connection, lote_id: str, agora: float) -> None:
    conexao.execute(
        "UPDATE lotes SET fim = ? WHERE id = ? AND fim IS NULL",
        (agora, lote_id)
    )
