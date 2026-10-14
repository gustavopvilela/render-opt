"""
Os dois datasets de treino: medições públicas + atributos das cenas + produção.

    python -m predict.dataset                 # usa data/processed/producao.parquet se existir
    python -m predict.dataset --sem-producao  # só os dados públicos

treino_vazao.parquet   alvo log_vazao
treino_custo.parquet   alvo log_custo, uma linha por ETAPA:
                         carregar            OB (medido ou deduzido) e produção
                         preparar_finalizar  Open Data e produção
Valor imputado nunca vira alvo: o 'carregar' imputado do OB e o do Open Data ficam de fora.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from predict import esquema

RAIZ = Path(__file__).resolve().parents[2]
PUBLICAS = RAIZ / "data" / "processed" / "medicoes_publicas.parquet"
ATRIBUTOS = RAIZ / "data" / "raw" / "atributos_cenas.csv"
PRODUCAO = RAIZ / "data" / "processed" / "producao.parquet"
SAIDA_VAZAO = RAIZ / "data" / "processed" / "treino_vazao.parquet"
SAIDA_CUSTO = RAIZ / "data" / "processed" / "treino_custo.parquet"


def atributos_de_cena (caminho: Path = ATRIBUTOS) -> pd.DataFrame:
    """Só as colunas de cena que os modelos usam (mais o trabalho da fórmula), uma linha por cena."""
    a = pd.read_csv(caminho)
    if faltam := [c for c in esquema.COLUNAS_CENA if c not in a.columns]:
        print(f"[aviso] colunas de cena ausentes no CSV (viram NaN): {faltam}")
    return a[["cena", "cena_sha256"] + [c for c in esquema.COLUNAS_CENA if c in a.columns]]


def publicas (caminho: Path, atributos: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_parquet(caminho)
    d = d.drop(columns=[c for c in atributos.columns if c != "cena" and c in d.columns])
    if sem := sorted(set(d["cena"]) - set(atributos["cena"])):
        print(f"[aviso] cenas sem atributos, linhas descartadas: {sem}")
    d = d.merge(atributos, on="cena", how="inner", validate="many_to_one")
    d["peso"] = 1.0
    return d


def producao (caminho: Path) -> pd.DataFrame | None:
    if not caminho.exists():
        return None
    p = pd.read_parquet(caminho)
    # Peso: cada (trabalhador, cena) soma 1
    p["peso"] = 1.0 / p.groupby(["trabalhador_id", "cena_sha256"])["frame"].transform("size")
    return p


def montar_vazao (pub: pd.DataFrame, prod: pd.DataFrame | None) -> pd.DataFrame:
    partes = [pub[pub["log_vazao"].notna()]]
    if prod is not None:
        partes.append(prod[prod["log_vazao"].notna()])
    return pd.concat(partes, ignore_index=True, sort=False)


def montar_custo (pub: pd.DataFrame, prod: pd.DataFrame | None) -> pd.DataFrame:
    ob = pub[(pub["fonte"] == "openbenchmarking") & (pub["carregamento_origem"] == "medido")]
    od = pub[(pub["fonte"] == "opendata") & pub["log_custo_fixo"].notna()]
    partes = [ob.assign(etapa="carregar", log_custo=np.log(ob["carregamento_s"])),
              od.assign(etapa="preparar_finalizar", log_custo=od["log_custo_fixo"])]
    if prod is not None:
        partes += [prod.assign(etapa="carregar", log_custo=prod["log_carregar"]),
                   prod.assign(etapa="preparar_finalizar", log_custo=prod["log_preparar"])]
    c = pd.concat(partes, ignore_index=True, sort=False)
    return c[np.isfinite(c["log_custo"])].reset_index(drop=True)


def resumo (nome: str, d: pd.DataFrame, por: list[str]) -> None:
    print(f"\n{nome}: {len(d):,} linhas, {d['maquina'].nunique():,} máquinas, {d['cena_sha256'].nunique()} cenas")
    print(d.groupby(por).agg(linhas=("peso", "size"), peso=("peso", "sum"),
                             cenas=("cena_sha256", "nunique")).round(1).to_string())


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publicas", type=Path, default=PUBLICAS)
    ap.add_argument("--atributos", type=Path, default=ATRIBUTOS)
    ap.add_argument("--producao", type=Path, default=PRODUCAO)
    ap.add_argument("--sem-producao", action="store_true")
    args = ap.parse_args()

    # Medições públicas com os atributos de cena e produção
    pub = publicas(args.publicas, atributos_de_cena(args.atributos))
    prod = None if args.sem_producao else producao(args.producao)
    print("produção:", "não usada" if prod is None else f"{len(prod):,} relatórios")

    # Datasets de vazão e de custo fixo
    vazao, custo = montar_vazao(pub, prod), montar_custo(pub, prod)
    vazao.to_parquet(SAIDA_VAZAO, index=False)
    custo.to_parquet(SAIDA_CUSTO, index=False)
    resumo("treino_vazao", vazao, ["fonte"])
    resumo("treino_custo", custo, ["etapa", "fonte"])
    print(f"\nescrito: {SAIDA_VAZAO}\n         {SAIDA_CUSTO}")


if __name__ == "__main__":
    main()
