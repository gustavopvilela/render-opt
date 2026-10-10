from __future__ import annotations
import sqlite3
from typing import Optional
from master.enums import EstadoTrabalhador

def buscar (conexao: sqlite3.Connection, trabalhador_id: str) -> Optional[sqlite3.Row]:
    return conexao.execute(
        "SELECT * FROM trabalhadores WHERE id = ?",
        (trabalhador_id,)
    ).fetchone()

def buscar_estado (conexao: sqlite3.Connection, trabalhador_id: str) -> Optional[sqlite3.Row]:
    return conexao.execute(
        "SELECT estado FROM trabalhadores WHERE id = ?",
        (trabalhador_id,)
    ).fetchone()

def listar (conexao: sqlite3.Connection) -> list[sqlite3.Row]:
    return conexao.execute(
        "SELECT id, dispositivo, estado, ultimo_heartbeat FROM trabalhadores"
    ).fetchall()

def disponiveis (conexao: sqlite3.Connection) -> list[sqlite3.Row]:
    return conexao.execute(
        "SELECT * FROM trabalhadores WHERE estado = ? ORDER BY id",
        (EstadoTrabalhador.DISPONIVEL,)
    ).fetchall()

def maquinas_disponiveis (conexao: sqlite3.Connection) -> list[str]:
    linhas = conexao.execute(
        "SELECT id FROM trabalhadores WHERE estado = ? ORDER BY id",
        (EstadoTrabalhador.DISPONIVEL,)
    )
    return [m for (m,) in linhas]

def silenciosos (conexao: sqlite3.Connection, antes_de: float) -> list[str]:
    return [m for (m,) in conexao.execute(
        "SELECT id FROM trabalhadores WHERE estado = ? AND ultimo_heartbeat < ?",
        (EstadoTrabalhador.DISPONIVEL, antes_de)
    )]

def salvar_registro (
    conexao: sqlite3.Connection,
    trabalhador_id: str,
    hardware: str,
    versoes_blender: str,
    dispositivo: str,
    agora: float
) -> None:
    conexao.execute(
        """
        INSERT INTO trabalhadores (id, hardware, versoes_blender, dispositivo, estado, ultimo_heartbeat)
        VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO
        UPDATE SET
            hardware = excluded.hardware,
            versoes_blender = excluded.versoes_blender,
            dispositivo = excluded.dispositivo,
            estado = excluded.estado,
            ultimo_heartbeat = excluded.ultimo_heartbeat
        """,
        (
            trabalhador_id,
            hardware,
            versoes_blender,
            dispositivo,
            EstadoTrabalhador.DISPONIVEL,
            agora
        )
    )

def marcar_estado (conexao: sqlite3.Connection, trabalhador_id: str, estado: EstadoTrabalhador) -> None:
    conexao.execute(
        "UPDATE trabalhadores SET estado = ? WHERE id = ?",
        (estado, trabalhador_id)
    )

def renovar_heartbeat_dos_disponiveis (conexao: sqlite3.Connection, agora: float) -> None:
    conexao.execute(
        "UPDATE trabalhadores SET ultimo_heartbeat = ? WHERE estado = ?",
        (agora, EstadoTrabalhador.DISPONIVEL)
    )

def renovar_heartbeat_se_disponivel (conexao: sqlite3.Connection, trabalhador_id: str, agora: float) -> int:
    cursor = conexao.execute(
        "UPDATE trabalhadores SET ultimo_heartbeat = ? WHERE id = ? AND estado = ?",
        (agora, trabalhador_id, EstadoTrabalhador.DISPONIVEL)
    )
    return cursor.rowcount

def renovar_heartbeat (conexao: sqlite3.Connection, trabalhador_id: str, agora: float) -> None:
    conexao.execute(
        "UPDATE trabalhadores SET ultimo_heartbeat = ? WHERE id = ?",
        (agora, trabalhador_id)
    )
