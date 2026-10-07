"""
Estado do mestre em SQLite.
As filas de cada máquina são descritas com um job PENDENTE, por exemplo,
com maquina = 'pc-01' e 'posicao = 3', está na posição 3 da fila do pc-01.
"""

from __future__ import annotations
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

ESQUEMA = """
CREATE TABLE IF NOT EXISTS trabalhadores (
    id                  TEXT PRIMARY KEY,
    hardware            TEXT NOT NULL,
    versoes_blender     TEXT NOT NULL,
    dispositivo         TEXT NOT NULL,
    estado              TEXT NOT NULL,      -- DISPONIVEL | INDISPONIVEL
    ultimo_heartbeat    REAL NOT NULL
);
    
CREATE TABLE IF NOT EXISTS lotes (
    id TEXT PRIMARY KEY,
    cena_id TEXT NOT NULL,
    cena_sha256 TEXT NOT NULL,
    versao_cena TEXT NOT NULL,
    criado_em REAL NOT NULL,
    inicio REAL, -- primeiro job entregue
    fim REAL -- ultimo job concluído
);
    
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    lote_id TEXT NOT NULL REFERENCES lotes(id),
    frame INTEGER NOT NULL,
    estado TEXT NOT NULL, -- PENDENTE | EXECUTANDO | CONCLUIDO | FALHA_DEFINITIVA
    maquina TEXT, -- dona da fila (PENDENTE) ou quem executa (EXECUTANDO)
    posicao REAL,
    token TEXT,
    inicio REAL,
    tentativas INTEGER NOT NULL DEFAULT 0,
    adiamentos INTEGER NOT NULL DEFAULT 0,
    evitar TEXT, -- última máquina que abortou/falhou este job
    tempo_real_s REAL
);
CREATE INDEX IF NOT EXISTS ix_fila ON jobs(estado, maquina, posicao);
    
CREATE TABLE IF NOT EXISTS estimativas (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    maquina TEXT NOT NULL,
    segundos REAL NOT NULL,
    PRIMARY KEY (job_id, maquina)
);
    
CREATE TABLE IF NOT EXISTS eventos (
    id INTEGER PRIMARY KEY,
    t REAL NOT NULL,
    tipo TEXT NOT NULL,
    job_id TEXT,
    maquina TEXT,
    dados TEXT
);
"""

class DB:
    def __init__(self, caminho: Path):
        caminho.parent.mkdir(parents=True, exist_ok=True)
        self.conexao = sqlite3.connect(caminho, check_same_thread=False, isolation_level=None)
        self.conexao.row_factory = sqlite3.Row
        self.conexao.execute("PRAGMA journal_mode=WAL")
        self.conexao.execute("PRAGMA foreign_keys=ON")
        self.conexao.executescript(ESQUEMA)
        self._trava = threading.Lock()
        self.ultimo_evento = self._maior_evento()

    def _maior_evento(self) -> int:
        (n,) = self.conexao.execute("SELECT COALESCE(MAX(id, 0) FROM eventos").fetchone()
        return n

    @contextmanager
    def transacao(self) -> Iterator[sqlite3.Connection]:
        with self._trava:
            self.conexao.execute("BEGIN IMMEDIATE")
            try:
                yield self.conexao
            except BaseException:
                self.conexao.execute("ROLLBACK")
                raise
            self.conexao.execute("COMMIT")
            self.ultimo_evento = self._maior_evento()

def fila (conexao: sqlite3.Connection, maquina: str) -> list[sqlite3.Row]:
    return conexao.execute(
        "SELECT * FROM jobs WHERE estado = 'PENDENTE' AND maquina = ? ORDER BY posicao",
        (maquina,),
    ).fetchall()

def colocar_no_fim (conexao: sqlite3.Connection, job_id: str, maquina: str) -> None:
    (ultima,) = conexao.execute(
        "SELECT COALESCE(MAX(posicao), 0) FROM jobs WHERE estado = 'PENDENTE' AND maquina = ?",
        (maquina,),
    ).fetchone()
    conexao.execute(
        "UPDATE jobs SET estado = 'PENDENTE', maquina = ?, posicao = ?, token = NULL, inicio = NULL WHERE id = ?",
        (maquina, ultima + 1, job_id)
    )

def estimativas_do_job (conexao: sqlite3.Connection, job_id: str) -> dict[str, float]:
    linhas = conexao.execute(
        "SELECT maquina, segundos FROM estimativas WHERE job_id = ?",
        (job_id,)
    )
    return {m: s for m, s in linhas}

def maquinas_disponiveis (conexao: sqlite3.Connection) -> list[str]:
    linhas = conexao.execute(
        "SELECT id FROM trabalhadores WHERE estado = 'DISPONIVEL' ORDER BY id"
    )
    return [m for (m,) in linhas]

def carga_pendente (conexao: sqlite3.Connection, maquina: str) -> float:
    (total,) = conexao.execute(
        "SELECT COALESCE(SUM(e.sugendos), 0) FROM jobs j "
        "JOIN estimativas e ON e.job_id = j.id AND e.maquina = j.maquina "
        "WHERE j.estado = 'PENDENTE' AND j.maquina = ?",
        (maquina,),
    ).fetchone()
    return total

def eventos_depois (conexao: sqlite3.Connection, id_evento: int, limite: int = 500) -> list[sqlite3.Row]:
    return conexao.execute(
        "SELECT * FROM eventos WHERE id > ? ORDER BY id LIMIT ?",
        (id_evento, limite),
    ).fetchall()

def registrar_evento(
    conexao: sqlite3.Connection,
    t: float,
    tipo: str,
    job_id: Optional[str] = None,
    maquina: Optional[str] = None,
    dados: Optional[str] = None
) -> None:
    conexao.execute(
        "INSERT INTO eventos (t, tipo, job_id, maquina, dados) VALUES (?, ?, ?, ?, ?)",
        (t, tipo, job_id, maquina, dados),
    )