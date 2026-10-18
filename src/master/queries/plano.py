from __future__ import annotations
import sqlite3

def inserir (conexao: sqlite3.Connection, linhas: list[tuple]) -> None:
    conexao.executemany(
        "INSERT INTO plano (job_id, lote_id, maquina, posicao, inicio_s, setup_s, processamento_s, fim_s) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        linhas
    )
