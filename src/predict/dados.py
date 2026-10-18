from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from predict import esquema, validacao

RAIZ = Path(__file__).resolve().parents[2]
DATASET = {
    "vazao": RAIZ / "data" / "processed" / "treino_vazao.parquet",
    "custo_fixo": RAIZ / "data" / "processed" / "treino_custo.parquet",
}
PASTA_EXPERIMENTOS = RAIZ / "experiments" / "predict"
ALVOS = tuple(DATASET)


def carregar (alvo: str, caminho: Optional[Path] = None) -> pd.DataFrame:
    df = pd.read_parquet(caminho or DATASET[alvo])
    df = df[df[esquema.COLUNA_ALVO[alvo]].notna()].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"Nenhuma linha com {esquema.COLUNA_ALVO[alvo]} em {caminho or DATASET[alvo]}")
    if "peso" not in df.columns:
        df["peso"] = 1.0
    return df


@dataclass
class Conjunto:
    df: pd.DataFrame
    cols: tuple[str, ...]
    X: pd.DataFrame
    categorias: dict
    y: np.ndarray
    w: np.ndarray

    def grupos (self, esquema_cv: str) -> np.ndarray:
        return validacao.grupos_de(self.df, esquema_cv)

    def outro (self, df: pd.DataFrame, alvo: str) -> "Conjunto":
        X, _ = esquema.montar_X(df, self.cols, self.categorias)
        return Conjunto(df, self.cols, X, self.categorias,
                        df[esquema.COLUNA_ALVO[alvo]].to_numpy(float), df["peso"].to_numpy(float))


def conjunto (df: pd.DataFrame, alvo: str) -> Conjunto:
    df = df.reset_index(drop=True)
    cols = esquema.sem_redundancia(df, esquema.colunas(alvo))
    X, cats = esquema.montar_X(df, cols)
    return Conjunto(df, cols, X, cats, df[esquema.COLUNA_ALVO[alvo]].to_numpy(float), df["peso"].to_numpy(float))


def separar (alvo: str, caminho: Optional[Path] = None) -> tuple[Conjunto, Conjunto]:
    df = carregar(alvo, caminho)
    dev, teste = validacao.teste_blindado(df)
    d = conjunto(df.iloc[dev], alvo)
    return d, d.outro(df.iloc[teste].reset_index(drop=True), alvo)
