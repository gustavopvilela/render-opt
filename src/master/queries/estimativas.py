from __future__ import annotations
import sqlite3
from master.enums import EstadoJob

def estimativas_do_job (conexao: sqlite3.Connection, job_id: str) -> dict[str, float]:
    linhas = conexao.execute(
        "SELECT maquina, segundos FROM estimativas WHERE job_id = ?",
        (job_id,)
    )
    return {m: s for m, s in linhas}

def carga_pendente (conexao: sqlite3.Connection, maquina: str) -> float:
    (total,) = conexao.execute(
        "SELECT COALESCE(SUM(e.sugendos), 0) FROM jobs j "
        "JOIN estimativas e ON e.job_id = j.id AND e.maquina = j.maquina "
        "WHERE j.estado = ? AND j.maquina = ?",
        (EstadoJob.PENDENTE, maquina),
    ).fetchone()
    return total

def multiplicar_segundos (conexao: sqlite3.Connection, job_id: str, fator: float) -> None:
    conexao.execute(
        "UPDATE jobs SET segundos = segundos * ? WHERE job_id = ?",
        (fator, job_id)
    )

def inserir (conexao: sqlite3.Connection, linhas: list[tuple[str, str, float]]) -> None:
    conexao.executemany(
        "INSERT INTO estimativas (job_id, maquina, segundos) VALUES (?, ?, ?)",
        linhas
    )

def inserir_se_ausentes (conexao: sqlite3.Connection, linhas: list[tuple[str, str, float]]) -> None:
    conexao.executemany(
        "INSERT OR IGNORE INTO estimativas (job_id, maquina, segundos) VALUES (?, ?, ?)",
        linhas
    )
