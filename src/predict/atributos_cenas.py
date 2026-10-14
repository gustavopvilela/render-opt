"""
Atributos das cenas públicas, lidos pelo MESMO extrator e pelo MESMO caminho que o mestre usa.

    python -m predict.atributos_cenas --blender "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe" \
        --cena classroom=../scenes/classroom/classroom/main.blend \
        --cena junkshop=../scenes/junkshop/junkshop/main.blend \
        --cena monster=../scenes/monster/monster/main.blend \
        --cena bmw27=D:/pts/bmw27_cpu.blend \
        --cena classroom_pts=D:/pts/isolado/classroom_cpu.blend \
        --cena fishy_cat=D:/pts/fishy_cat_cpu.blend \
        --cena pabellon=D:/pts/pavillon_barcelone_cpu.blend \
        --cena barbershop=D:/pts/barbershop_interior_cpu.blend

Saída: data/raw/atributos_cenas.csv, uma linha por cena (frame 1), com 'cena',
'cena_sha256' e todas as colunas do extrator. Os nomes das cenas são os do
dataset (opendata.py e openbenchmarking.CENA).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from comum.util import sha256_arquivo
from predict import features_cena

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "data" / "raw" / "atributos_cenas.csv"
CACHE = RAIZ / "data" / "raw" / "cache_atributos"
CENAS_ESPERADAS = ("barbershop", "bmw27", "classroom", "classroom_pts", "fishy_cat", "junkshop", "monster", "pabellon")
FRAME = 1


def extrair_cena (blender: str, nome: str, blend: Path, pasta_cache: Path) -> dict:
    linha = features_cena.extrair(blender, blend, [FRAME], pasta_cache).iloc[0].to_dict()
    # O nome do dataset substitui o nome interno da cena que o extrator devolve
    return {**linha, "cena": nome, "cena_sha256": sha256_arquivo(blend)}


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blender", required=True)
    ap.add_argument("--cena", action="append", required=True, metavar="NOME=ARQUIVO.blend")
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args()

    linhas = []
    for item in args.cena:
        nome, _, caminho = item.partition("=")
        blend = Path(caminho).resolve()
        if not blend.is_file():
            raise SystemExit(f"{nome}: arquivo não encontrado: {blend}")
        linhas.append(extrair_cena(args.blender, nome.strip(), blend, args.cache))
        r = linhas[-1]
        print(f"{r['cena']:14s} {r['cena_sha256'][:12]}  amostras {r.get('amostras')}  pixels {r.get('pixels_totais')}", flush=True)

    df = pd.DataFrame(linhas)
    if repetidas := sorted(df.loc[df["cena"].duplicated(), "cena"]):
        raise SystemExit(f"Cena informada duas vezes: {repetidas}")
    if faltam := sorted(set(CENAS_ESPERADAS) - set(df["cena"])):
        print(f"[aviso] faltam as cenas {faltam}: as linhas delas ficarão de fora do treino")
    numericas = df.select_dtypes("number")
    print(f"colunas numéricas que variam entre as cenas: {int((numericas.nunique() > 1).sum())} de {numericas.shape[1]}")
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False)
    print(f"escrito: {args.saida}")


if __name__ == "__main__":
    main()
