"""Onde ficam os datasets de cada alvo e como carregá-los."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from predict.esquema import COLUNA_ALVO

RAIZ = Path(__file__).resolve().parents[2]
DATASET = {
    "vazao": RAIZ / "data" / "processed" / "treino_vazao.parquet",
    "custo_fixo": RAIZ / "data" / "processed" / "treino_custo.parquet",
}
PASTA_EXPERIMENTOS = RAIZ / "experiments" / "predict"


def carregar (alvo: str, caminho: Optional[Path] = None) -> pd.DataFrame:
    """Dataset do alvo, só com as linhas que têm o alvo."""
    df = pd.read_parquet(caminho or DATASET[alvo])
    df = df[df[COLUNA_ALVO[alvo]].notna()].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"Nenhuma linha com {COLUNA_ALVO[alvo]} em {caminho or DATASET[alvo]}")
    if "peso" not in df.columns:
        df["peso"] = 1.0
    return df
