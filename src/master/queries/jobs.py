from __future__ import annotations
import sqlite3
from typing import Optional
from master.enums import EstadoJob

def buscar (conexao: sqlite3.Connection, job_id: str) -> Optional[sqlite3.Row]:
    return conexao.execute(
        "SELECT * FROM jobs WHERE id = ?",
        (job_id,)
    ).fetchone()

def evitar_do_job (conexao: sqlite3.Connection, job_id: str) -> Optional[str]:
    (evitar,) = conexao.execute(
        "SELECT evitar FROM jobs WHERE id = ?",
        (job_id,)
    ).fetchone()
    return evitar

def fila (conexao: sqlite3.Connection, maquina: str) -> list[sqlite3.Row]:
    return conexao.execute(
        "SELECT * FROM jobs WHERE estado = ? AND maquina = ? ORDER BY posicao",
        (EstadoJob.PENDENTE, maquina),
    ).fetchall()

def primeiro_da_fila (conexao: sqlite3.Connection, maquina: str) -> Optional[sqlite3.Row]:
    return conexao.execute(
        "SELECT * FROM jobs WHERE estado = ? AND maquina = ? ORDER BY posicao LIMIT 1",
        (EstadoJob.PENDENTE, maquina)
    ).fetchone()

def orfao_elegivel (conexao: sqlite3.Connection, maquina: str) -> Optional[sqlite3.Row]:
    return conexao.execute(
        "SELECT j.* FROM jobs j JOIN estimativas e ON e.job_id = j.id AND e.maquina = ? "
        "WHERE j.estado = ? AND j.maquina IS NULL ORDER BY j.id LIMIT 1",
        (maquina, EstadoJob.PENDENTE)
    ).fetchone()

def executando_na_maquina (conexao: sqlite3.Connection, maquina: str) -> list[str]:
    return [job_id for (job_id,) in conexao.execute(
        "SELECT id FROM jobs WHERE estado = ? AND maquina = ?",
        (EstadoJob.EXECUTANDO, maquina)
    ).fetchall()]

def pendentes_sem_estimativa (conexao: sqlite3.Connection, maquina: str) -> list[sqlite3.Row]:
    return conexao.execute(
        "SELECT j.id, j.lote_id, j.frame FROM jobs j WHERE j.estado = ? AND NOT EXISTS "
        "(SELECT 1 FROM estimativas e WHERE e.job_id = j.id AND e.maquinas = ?) ORDER BY j.lote_id",
        (EstadoJob.PENDENTE, maquina)
    ).fetchall()

def restantes_do_lote (conexao: sqlite3.Connection, lote_id: str) -> int:
    (restantes,) = conexao.execute(
        "SELECT COUNT(*) FROM jobs WHERE lote_id = ? AND estado IN (?, ?)",
        (lote_id, EstadoJob.PENDENTE, EstadoJob.EXECUTANDO)
    ).fetchone()
    return restantes

def inserir (conexao: sqlite3.Connection, job_id: str, lote_id: str, frame: int, maquina: str, posicao: int) -> None:
    conexao.execute(
        "INSERT INTO jobs (id, lote_id, frame, estado, maquina, posicao) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, lote_id, frame, EstadoJob.PENDENTE, maquina, posicao)
    )

def colocar_no_fim (conexao: sqlite3.Connection, job_id: str, maquina: str) -> None:
    (ultima,) = conexao.execute(
        "SELECT COALESCE(MAX(posicao), 0) FROM jobs WHERE estado = ? AND maquina = ?",
        (EstadoJob.PENDENTE, maquina),
    ).fetchone()
    conexao.execute(
        "UPDATE jobs SET estado = ?, maquina = ?, posicao = ?, token = NULL, inicio = NULL WHERE id = ?",
        (EstadoJob.PENDENTE, maquina, ultima + 1, job_id)
    )

def tornar_orfao (conexao: sqlite3.Connection, job_id: str) -> None:
    conexao.execute(
        "UPDATE jobs SET estado = ?, maquina = NULL, posicao = NULL, "
        "token = NULL, inicio = NULL WHERE id = ?",
        (EstadoJob.PENDENTE, job_id)
    )

def atualizar_tentativas (conexao: sqlite3.Connection, job_id: str, tentativas: int, evitar: str) -> None:
    conexao.execute(
        "UPDATE jobs SET tentativas = ?, evitar = ? WHERE id = ?",
        (tentativas, evitar, job_id)
    )

def registrar_adiamento (conexao: sqlite3.Connection, job_id: str, evitar: str) -> None:
    conexao.execute(
        "UPDATE jobs SET adiamentos = adiamentos + 1, evitar = ? WHERE id = ?",
        (evitar, job_id)
    )

def marcar_executando (conexao: sqlite3.Connection, job_id: str, maquina: str, token: str, inicio: float) -> None:
    conexao.execute(
        "UPDATE jobs SET estado = ?, maquina = ?, posicao = NULL, token = ?, inicio = ? WHERE id = ?",
        (EstadoJob.EXECUTANDO, maquina, token, inicio, job_id)
    )

def marcar_concluido (conexao: sqlite3.Connection, job_id: str, tempo_real_s: float) -> None:
    conexao.execute(
        "UPDATE jobs SET estado = ?, token = NULL, tempo_real_s = ? WHERE id = ?",
        (EstadoJob.CONCLUIDO, tempo_real_s, job_id)
    )

def marcar_falha_definitiva (conexao: sqlite3.Connection, job_id: str) -> None:
    conexao.execute(
        "UPDATE jobs SET estado = ?, token = NULL, posicao = NULL WHERE id = ?",
        (EstadoJob.FALHA_DEFINITIVA, job_id)
    )

def contagem_por_estado (conexao: sqlite3.Connection, lote_id: str) -> dict[str, int]:
    return dict(conexao.execute(
        "SELECT estado, COUNT(*) FROM jobs WHERE lote_id = ? GROUP BY estado",
        (lote_id,)
    ).fetchall())

def concluidos_por_maquina (conexao: sqlite3.Connection, lote_id: str) -> sqlite3.Cursor:
    return conexao.execute(
        "SELECT maquina, COUNT(*), COALESCE(SUM(tempo_real_s), 0) FROM jobs "
        "WHERE lote_id = ? AND estado = ? GROUP BY maquina",
        (lote_id, EstadoJob.CONCLUIDO)
    )

def pendentes_por_maquina (conexao: sqlite3.Connection, lote_id: str) -> sqlite3.Cursor:
    return conexao.execute(
        "SELECT maquina, COUNT(*) FROM jobs WHERE lote_id = ? AND estado = ? GROUP BY maquina",
        (lote_id, EstadoJob.PENDENTE)
    )

def executando_por_maquina (conexao: sqlite3.Connection, lote_id: str) -> sqlite3.Cursor:
    return conexao.execute(
        "SELECT maquina, id FROM jobs WHERE lote_id = ? AND estado = ?",
        (lote_id, EstadoJob.EXECUTANDO)
    )
