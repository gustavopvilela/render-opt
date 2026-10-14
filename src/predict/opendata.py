"""
O Blender Open Data agregado por (máquina, cena, versão maior do Blender).

Lê o CSV sem agregação gerado por scripts/montar_dataset_opendata.py (uma linha
por submissão) e produz uma linha por grupo, com a mediana dos alvos.

    python -m predict.opendata --min-medicoes 10

Por que a versão do Blender entra na chave: a mesma placa ficou muito mais
rápida entre o Blender 3 e o 5. Misturar as versões numa mediana só mistura
desempenhos diferentes, e a versão vira uma feature sem significado.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from predict.hardware import features_hardware

RAIZ = Path(__file__).resolve().parents[2]
CSV_SEM_AGREGAR = RAIZ / "data" / "processed" / "dataset_opendata_sem_agregar.csv"
SAIDA = RAIZ / "data" / "processed" / "opendata.parquet"

CHAVE = ["maquina", "cena", "versao_blender_major"]
COLUNAS = [
    "cena", "maquina", "dispositivo_nome", "dispositivo_tipo", "cpu_nome",
    "num_cpu_cores", "num_cpu_threads", "num_cpu_sockets", "sistema",
    "versao_blender_major", "versao_blender_minor",
    "vazao_amostras_pixel_s", "custo_fixo_s", "medido_num_amostras",
]
SENTINELA_AMOSTRAS = 2 ** 24
CORTE_CUSTO_FIXO = 20.0


def _moda (s: pd.Series):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else None


def carregar (caminho: Path = CSV_SEM_AGREGAR) -> pd.DataFrame:
    df = pd.read_csv(caminho, usecols=COLUNAS)
    antes = len(df)
    df = df[df["medido_num_amostras"] < SENTINELA_AMOSTRAS]
    print(f"submissões: {antes:,}; sentinelas descartadas: {antes - len(df)}")
    return df


def agregar (df: pd.DataFrame, min_medicoes: int) -> pd.DataFrame:
    g = df.groupby(CHAVE, sort=False)
    out = g.agg(
        n_medicoes=("vazao_amostras_pixel_s", "size"),
        vazao_amostras_pixel_s=("vazao_amostras_pixel_s", "median"),
        custo_fixo_s=("custo_fixo_s", "median"),
        dispositivo_nome=("dispositivo_nome", "first"),
        dispositivo_tipo=("dispositivo_tipo", "first"),
        num_cpu_cores=("num_cpu_cores", "median"),
        num_cpu_threads=("num_cpu_threads", "median"),
        num_cpu_sockets=("num_cpu_sockets", "median"),
        versao_blender_minor=("versao_blender_minor", "median"),
    ).reset_index()
    out["sistema"] = g["sistema"].agg(_moda).values
    out["cpu_nome"] = g["cpu_nome"].agg(_moda).values
    out = out[out["n_medicoes"] >= min_medicoes].reset_index(drop=True)

    out["log_vazao"] = np.log(out["vazao_amostras_pixel_s"])

    # Custo fixo acima de 20x a mediana da cena vira NaN
    mediana_cena = out.groupby("cena")["custo_fixo_s"].transform("median")
    estourado = out["custo_fixo_s"] > CORTE_CUSTO_FIXO * mediana_cena
    print(f"custo fixo acima de {CORTE_CUSTO_FIXO:.0f}x a mediana da cena: {int(estourado.sum())} linhas -> NaN")
    out.loc[estourado | (out["custo_fixo_s"] <= 0), "custo_fixo_s"] = np.nan
    out["log_custo_fixo"] = np.log(out["custo_fixo_s"])
    return out


def acrescentar_hardware (df: pd.DataFrame) -> pd.DataFrame:
    hw = pd.DataFrame([features_hardware(n, t) for n, t in zip(df["dispositivo_nome"], df["dispositivo_tipo"])])
    # CPU hospedeira
    host = pd.DataFrame([features_hardware(n, "CPU") for n in df["cpu_nome"]])
    host = host[["hw_familia", "hw_ano", "hw_classe"]].add_prefix("host_")
    host.columns = [c.replace("host_hw_", "host_") for c in host.columns]
    return pd.concat([df.reset_index(drop=True), hw, host], axis=1)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", type=Path, default=CSV_SEM_AGREGAR)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--min-medicoes", type=int, default=10)
    args = ap.parse_args()

    # Agregação e atributos de hardware
    df = acrescentar_hardware(agregar(carregar(args.entrada), args.min_medicoes))
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.saida, index=False)

    print(f"\nlinhas: {len(df):,}   máquinas: {df['maquina'].nunique():,}")
    print("por tipo:", df["dispositivo_tipo"].value_counts().to_dict())
    print("por versão:", df["versao_blender_major"].value_counts().sort_index().to_dict())
    print(f"hw_ano preenchido: {df['hw_ano'].notna().mean():.1%}")
    print(f"escrito: {args.saida}")


if __name__ == "__main__":
    main()
