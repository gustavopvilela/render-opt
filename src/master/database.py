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
    criado_em REAL NOT NULL,
    inicio REAL, -- primeiro job entregue
    fim REAL, -- ultimo job concluído
    distribuicao TEXT NOT NULL DEFAULT 'sa' -- sa | flamenco
);
    
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    lote_id TEXT NOT NULL REFERENCES lotes(id),
    cena_id TEXT NOT NULL,
    cena_sha256 TEXT NOT NULL,
    versao_cena TEXT NOT NULL,
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
    
CREATE TABLE IF NOT EXISTS plano (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id),
    lote_id TEXT NOT NULL REFERENCES lotes(id),
    maquina TEXT NOT NULL,
    posicao INTEGER NOT NULL,
    inicio_s REAL NOT NULL,
    setup_s REAL NOT NULL,
    processamento_s REAL NOT NULL,
    fim_s REAL NOT NULL
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
        colunas = {linha["name"] for linha in self.conexao.execute("PRAGMA table_info(lotes)")}
        if "distribuicao" not in colunas:
            self.conexao.execute("ALTER TABLE lotes ADD COLUMN distribuicao TEXT NOT NULL DEFAULT 'sa'")
        self._trava = threading.Lock()
        self.ultimo_evento = self._maior_evento()

    def _maior_evento(self) -> int:
        (n,) = self.conexao.execute("SELECT COALESCE(MAX(id), 0) FROM eventos").fetchone()
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
