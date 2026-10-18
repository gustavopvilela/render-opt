from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from predict.opendata import CSV_SEM_AGREGAR, SENTINELA_AMOSTRAS

ALVO = {"vazao": "vazao_amostras_pixel_s", "custo_fixo": "custo_fixo_s"}
CHAVE = ["maquina", "cena", "versao_blender_major"]
CORTE_LOG = np.log(20.0)


def calcular (caminho: Path, alvo: str, min_medicoes: int) -> dict[str, float]:
    col = ALVO[alvo]
    df = pd.read_csv(caminho, usecols=CHAVE + [col, "medido_num_amostras"])
    df = df[(df["medido_num_amostras"] < SENTINELA_AMOSTRAS) & (df[col] > 0)]
    df["log"] = np.log(df[col])
    tamanho = df.groupby(CHAVE)["log"].transform("size")
    df = df[tamanho >= min_medicoes]
    desvio = df["log"] - df.groupby(CHAVE)["log"].transform("median")
    quebrados = desvio.abs() > CORTE_LOG
    desvio = desvio[~quebrados]
    por_grupo = desvio.abs().groupby([df.loc[desvio.index, c] for c in CHAVE]).median()
    return {
        "grupos": int(len(por_grupo)),
        "submissoes": int(len(desvio)),
        "descartadas_quebradas": int(quebrados.sum()),
        "fator_mediano_ruido": float(np.exp(por_grupo.median())),
        "sigma_ruido_log": float(1.4826 * desvio.abs().median()),
    }


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", type=Path, default=CSV_SEM_AGREGAR)
    ap.add_argument("--alvo", choices=tuple(ALVO), default="vazao")
    ap.add_argument("--min-medicoes", type=int, default=10)
    args = ap.parse_args()
    for chave, valor in calcular(args.entrada, args.alvo, args.min_medicoes).items():
        print(f"{chave:24s} {valor:.4f}" if isinstance(valor, float) else f"{chave:24s} {valor}")


if __name__ == "__main__":
    main()
