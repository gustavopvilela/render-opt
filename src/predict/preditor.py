"""
Inferência: carrega o artefato do treinar.py e monta a matriz P (frames x máquinas).

    preditor = Preditor(Path("models/2026-10-20T15-02-11"))
    P = preditor.prever_matriz_p(frames, maquinas, cena_sha256=sha)                    # segundos
    P_cons = preditor.prever_matriz_p(frames, maquinas, cena_sha256=sha, conservador=True)

p_ij = c_carregar + c_preparar + S_i * W_i * H_i / v_ij, com c e v previstos e S, W, H lidos do .blend.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from predict import esquema, incerteza

LIMITE_S = 7 * 24 * 3600.0


class Preditor:
    def __init__ (self, pasta: Path):
        self.pasta = Path(pasta)
        self.meta = json.loads((self.pasta / "metadados.json").read_text(encoding="utf-8"))
        self.modelos = {
            (alvo, tipo): lgb.Booster(model_file=str(self.pasta / f"{alvo}_{tipo}.txt"))
            for alvo in ("vazao", "custo_fixo") for tipo in ("mediana", "quantil")
        }
        self.etapas = tuple(self.meta.get("etapas_custo", esquema.ETAPAS))

    def cena_conhecida (self, alvo: str, cena_sha256: Optional[str]) -> bool:
        return cena_sha256 is not None and cena_sha256 in set(self.meta["alvos"][alvo]["cenas"])

    def _prever (self, alvo: str, X_base: pd.DataFrame, conservador: bool, cena_nova: bool = False) -> np.ndarray:
        info = self.meta["alvos"][alvo]
        X, _ = esquema.montar_X(X_base, tuple(info["colunas"]), info["categorias"])
        if not conservador:
            return self.modelos[(alvo, "mediana")].predict(X)
        folga = info["folga_cena_nova_log"] if cena_nova else info["folga_conformal_log"]
        return incerteza.aplicar(self.modelos[(alvo, "quantil")].predict(X), folga, alvo)

    def prever_matriz_p (self, frames: pd.DataFrame, maquinas: pd.DataFrame,
                         cena_sha256: Optional[str] = None, conservador: bool = False) -> np.ndarray:
        """
        :param frames: uma linha por frame, com as colunas do extrator (inclui amostras e pixels_totais)
        :param maquinas: uma linha por máquina, com as colunas de hardware.linha_maquina()
        :param cena_sha256: decide a folga do limite conservador (cena do treino ou cena nova)
        :param conservador: True usa o limite pessimista (vazão q10 e custo q90, com folga conformal)
        :return: matriz (n_frames, n_maquinas) em segundos
        """
        P, C = self.prever_matrizes(frames, maquinas, cena_sha256, conservador)
        return P + C

    def prever_matrizes (self, frames: pd.DataFrame, maquinas: pd.DataFrame,
                         cena_sha256: Optional[str] = None, conservador: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Mesmos parâmetros de prever_matriz_p.
        :return: (processamento, custo fixo), as duas (n_frames, n_maquinas) em segundos
        """
        n, m = len(frames), len(maquinas)
        # Colunas presentes nos dois lados pertencem à máquina
        frames = frames.drop(columns=[c for c in maquinas.columns if c in frames.columns])
        # Vazão e custo fixo (soma das etapas) para todos os pares frame x máquina
        pares = frames.reset_index(drop=True).merge(maquinas.reset_index(drop=True), how="cross")
        log_v = self._prever("vazao", pares, conservador, not self.cena_conhecida("vazao", cena_sha256))
        nova_c = not self.cena_conhecida("custo_fixo", cena_sha256)
        custo = sum(np.exp(self._prever("custo_fixo", pares.assign(etapa=e), conservador, nova_c))
                    for e in self.etapas)
        trabalho = pares["amostras"].to_numpy(float) * pares["pixels_totais"].to_numpy(float)
        P = (trabalho / np.exp(log_v)).reshape(n, m)
        C = np.asarray(custo, dtype=float).reshape(n, m)

        # Verificação de sanidade
        total = P + C
        ruins = ~np.isfinite(total) | (total <= 0) | (total > LIMITE_S)
        if ruins.any():
            i, j = map(int, np.argwhere(ruins)[0])
            raise ValueError(f"Previsão inválida para o frame da linha {i} na máquina da coluna {j}: {total[i, j]}")
        return P, C
