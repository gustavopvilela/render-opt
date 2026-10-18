from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from predict import especificacoes, esquema, incerteza

LIMITE_S = 7 * 24 * 3600.0


class Preditor:
    def __init__ (self, pasta: Path):
        self.pasta = Path(pasta)
        self.meta = json.loads((self.pasta / "metadados.json").read_text(encoding="utf-8"))
        self.modelos = {alvo: joblib.load(self.pasta / f"{alvo}.joblib") for alvo in self.meta["alvos"]}
        self.especificacoes = pd.read_parquet(self.pasta / "especificacoes.parquet")
        self.etapas = tuple(self.meta["etapas_custo"])

    def cena_conhecida (self, alvo: str, cena_sha256: Optional[str]) -> bool:
        return cena_sha256 is not None and cena_sha256 in set(self.meta["alvos"][alvo]["cenas"])

    def _prever (self, alvo: str, X_base: pd.DataFrame, conservador: bool, cena_nova: bool = False) -> np.ndarray:
        info = self.meta["alvos"][alvo]
        X, _ = esquema.montar_X(X_base, tuple(info["colunas"]), info["categorias"])
        p = self.modelos[alvo].prever(X)
        if not conservador:
            return p
        return incerteza.aplicar(p, info["folga_cena_nova_log"] if cena_nova else info["folga_maquina_log"], alvo)

    def prever_matriz_p (self, frames: pd.DataFrame, maquinas: pd.DataFrame,
                         cena_sha256: Optional[str] = None, conservador: bool = False) -> np.ndarray:
        P, C = self.prever_matrizes(frames, maquinas, cena_sha256, conservador)
        return P + C

    def prever_matrizes (self, frames: pd.DataFrame, maquinas: pd.DataFrame,
                         cena_sha256: Optional[str] = None, conservador: bool = False) -> tuple[np.ndarray, np.ndarray]:
        n, m = len(frames), len(maquinas)
        maquinas = especificacoes.acrescentar(maquinas.reset_index(drop=True), self.especificacoes)
        frames = frames.drop(columns=[c for c in maquinas.columns if c in frames.columns])
        pares = frames.reset_index(drop=True).merge(maquinas, how="cross")
        log_v = self._prever("vazao", pares, conservador, not self.cena_conhecida("vazao", cena_sha256))
        nova_c = not self.cena_conhecida("custo_fixo", cena_sha256)
        custo = sum(np.exp(self._prever("custo_fixo", pares.assign(etapa=e), conservador, nova_c)) for e in self.etapas)
        trabalho = pares["amostras"].to_numpy(float) * pares["pixels_totais"].to_numpy(float)
        P = (trabalho / np.exp(log_v)).reshape(n, m)
        C = np.asarray(custo, dtype=float).reshape(n, m)

        total = P + C
        ruins = ~np.isfinite(total) | (total <= 0) | (total > LIMITE_S)
        if ruins.any():
            i, j = map(int, np.argwhere(ruins)[0])
            raise ValueError(f"Previsão inválida para o frame da linha {i} na máquina da coluna {j}: {total[i, j]}")
        return P, C
