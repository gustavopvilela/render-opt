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
Z_MAX = 3.5
MIN_PARA_FILTRAR = 3


def _moda (s: pd.Series):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else None


def carregar (caminho: Path = CSV_SEM_AGREGAR) -> pd.DataFrame:
    df = pd.read_csv(caminho, usecols=COLUNAS)
    antes = len(df)
    df = df[(df["medido_num_amostras"] < SENTINELA_AMOSTRAS) & (df["vazao_amostras_pixel_s"] > 0)].copy()
    print(f"submissões: {antes:,}; descartadas (sentinela ou vazão <= 0): {antes - len(df)}")
    df.loc[df["custo_fixo_s"] <= 0, "custo_fixo_s"] = np.nan
    return df


def atipicos (log: pd.Series, grupos: list[pd.Series]) -> pd.Series:
    g = log.groupby(grupos)
    mediana = g.transform("median")
    mad = (log - mediana).abs().groupby(grupos).transform("median")
    z = 0.6745 * (log - mediana) / mad.replace(0, np.nan)
    return (g.transform("count") >= MIN_PARA_FILTRAR) & (z.abs() > Z_MAX)


def agregar (df: pd.DataFrame) -> pd.DataFrame:
    grupos = [df[c] for c in CHAVE]
    lv, lc = np.log(df["vazao_amostras_pixel_s"]), np.log(df["custo_fixo_s"])
    fora_v, fora_c = atipicos(lv, grupos), atipicos(lc, grupos)
    print(f"atípicos removidos dentro do grupo: vazão {int(fora_v.sum()):,}, custo fixo {int(fora_c.sum()):,}")
    df = df.assign(log_vazao=lv.where(~fora_v), log_custo_fixo=lc.where(~fora_c))

    g = df.groupby(CHAVE, sort=False)
    out = g.agg(
        n_medicoes=("log_vazao", "count"),
        n_medicoes_custo=("log_custo_fixo", "count"),
        log_vazao=("log_vazao", "median"),
        log_custo_fixo=("log_custo_fixo", "median"),
        dispositivo_nome=("dispositivo_nome", "first"),
        dispositivo_tipo=("dispositivo_tipo", "first"),
        num_cpu_cores=("num_cpu_cores", "median"),
        num_cpu_threads=("num_cpu_threads", "median"),
        num_cpu_sockets=("num_cpu_sockets", "median"),
        versao_blender_minor=("versao_blender_minor", "median"),
    ).reset_index()
    out["sistema"] = g["sistema"].agg(_moda).values
    out["cpu_nome"] = g["cpu_nome"].agg(_moda).values
    out = out[out["n_medicoes"] > 0].reset_index(drop=True)
    out["vazao_amostras_pixel_s"] = np.exp(out["log_vazao"])
    out["custo_fixo_s"] = np.exp(out["log_custo_fixo"])

    estourado = out["custo_fixo_s"] > CORTE_CUSTO_FIXO * out.groupby("cena")["custo_fixo_s"].transform("median")
    print(f"custo fixo acima de {CORTE_CUSTO_FIXO:.0f}x a mediana da cena: {int(estourado.sum())} grupos -> NaN")
    out.loc[estourado, ["custo_fixo_s", "log_custo_fixo"]] = np.nan
    out.loc[estourado, "n_medicoes_custo"] = 0
    return out


def acrescentar_hardware (df: pd.DataFrame) -> pd.DataFrame:
    hw = pd.DataFrame([features_hardware(n, t) for n, t in zip(df["dispositivo_nome"], df["dispositivo_tipo"])])
    host = pd.DataFrame([features_hardware(n, "CPU") for n in df["cpu_nome"]])
    host = host[["hw_familia", "hw_ano", "hw_classe"]].add_prefix("host_")
    host.columns = [c.replace("host_hw_", "host_") for c in host.columns]
    return pd.concat([df.reset_index(drop=True), hw, host], axis=1)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", type=Path, default=CSV_SEM_AGREGAR)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    args = ap.parse_args()

    df = acrescentar_hardware(agregar(carregar(args.entrada)))
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.saida, index=False)

    n = df["n_medicoes"]
    print(f"\nlinhas: {len(df):,}   máquinas: {df['maquina'].nunique():,}")
    print(f"submissões por grupo: 1 -> {(n == 1).mean():.1%}, 2-9 -> {n.between(2, 9).mean():.1%}, >=10 -> {(n >= 10).mean():.1%}")
    print("por tipo:", df["dispositivo_tipo"].value_counts().to_dict())
    print("por versão:", df["versao_blender_major"].value_counts().sort_index().to_dict())
    print(f"escrito: {args.saida}")


if __name__ == "__main__":
    main()
