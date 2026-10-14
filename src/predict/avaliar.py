"""
Validação cruzada agrupada de vários modelos, com registro em CSV.

    python -m predict.avaliar --alvo vazao --esquema maquina \
        --modelos B0,B1,B2,ridge,floresta,lgbm --sementes 3
    python -m predict.avaliar --alvo vazao --esquema cena      # deixa uma cena de fora

    # uma única vez, no fim, com o modelo já escolhido:
    python -m predict.avaliar --alvo vazao --modelos B1,lgbm --teste

O teste blindado é sempre um conjunto de MÁQUINAS (20%), o mesmo em todos os scripts.
Toda avaliação vira linhas em experiments/predict/metricas.csv. Tabelas e
figuras da monografia saem desse arquivo, nunca de números copiados do terminal.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from predict import dados, esquema, metricas, modelos, validacao

CSV_METRICAS = dados.PASTA_EXPERIMENTOS / "metricas.csv"


def separar (df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Separa o teste blindado (20% das máquinas, pelo hash do nome) e grava quem ficou nele."""
    dev, teste = validacao.teste_blindado(df)
    dados.PASTA_EXPERIMENTOS.mkdir(parents=True, exist_ok=True)
    (dados.PASTA_EXPERIMENTOS / "teste_maquinas.txt").write_text(
        "\n".join(sorted(set(df["maquina"].iloc[teste].astype(str)))), encoding="utf-8")
    return dev, teste


def recortes_de (df: pd.DataFrame, esquema_cv: str) -> pd.Series:
    """Rótulo do recorte de cada linha: tipo de dispositivo; etapa no custo; a cena no esquema 'cena'."""
    r = "tipo:" + validacao.texto(df["dispositivo_tipo"])
    if "etapa" in df.columns:
        r = r + "|etapa:" + validacao.texto(df["etapa"])
    if esquema_cv == "cena":
        r = "cena:" + validacao.texto(df["cena"]) + ("|etapa:" + validacao.texto(df["etapa"]) if "etapa" in df.columns else "")
    return r.reset_index(drop=True)


def linhas_de_metricas (y, p, recortes: pd.Series, base: dict) -> list[dict]:
    """Uma linha para o conjunto todo e uma por recorte."""
    saida = [{**base, "recorte": "todos", **metricas.resumo(y, p)}]
    for rotulo, idx in recortes.groupby(recortes.values).groups.items():
        pos = recortes.index.get_indexer(idx)
        saida.append({**base, "recorte": str(rotulo), **metricas.resumo(y[pos], p[pos])})
    return saida


def avaliar (
    df: pd.DataFrame, alvo: str, nomes: list[str], esquema_cv: str,
    sementes: int, params: Optional[dict] = None, teste: bool = False,
) -> pd.DataFrame:
    # Separação do teste blindado
    cols = esquema.colunas(alvo)
    y_todo = df[esquema.COLUNA_ALVO[alvo]].to_numpy(float)
    w_todo = df["peso"].to_numpy(float)
    dev, idx_teste = separar(df)
    d = df.iloc[dev].reset_index(drop=True)
    X_dev, categorias = esquema.montar_X(d, cols)
    y_dev, w_dev = y_todo[dev], w_todo[dev]
    g_parada = validacao.grupos_de(d, "maquina")
    g_cv = validacao.grupos_de(d, esquema_cv)
    rec_dev = recortes_de(d, esquema_cv)

    linhas = []
    base = {"data": time.strftime("%Y-%m-%dT%H:%M:%S"), "alvo": alvo,
            "esquema": "teste" if teste else esquema_cv, "params": json.dumps(params or {})}

    # Avaliação única no teste blindado
    if teste:
        t = df.iloc[idx_teste].reset_index(drop=True)
        X_te, _ = esquema.montar_X(t, cols, categorias)
        for nome in nomes:
            m = modelos.criar(nome, alvo, params).ajustar(X_dev, y_dev, g_parada, w_dev)
            linhas += linhas_de_metricas(y_todo[idx_teste], m.prever(X_te), recortes_de(t, "maquina"),
                                         {**base, "modelo": nome, "semente": 0, "dobra": 0})
        return pd.DataFrame(linhas)

    # Validação cruzada agrupada
    n_sementes = 1 if esquema_cv == "cena" else sementes
    for semente in range(n_sementes):
        for k, (tr, va) in enumerate(validacao.dobras(g_cv, esquema_cv, 5, semente)):
            for nome in nomes:
                m = modelos.criar(nome, alvo, params, semente).ajustar(X_dev.iloc[tr], y_dev[tr], g_parada[tr], w_dev[tr])
                p = m.prever(X_dev.iloc[va])
                linhas += linhas_de_metricas(y_dev[va], p, rec_dev.iloc[va].reset_index(drop=True),
                                             {**base, "modelo": nome, "semente": semente, "dobra": k})
    return pd.DataFrame(linhas)


def registrar (resultado: pd.DataFrame, caminho: Path = CSV_METRICAS) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(caminho, mode="a", header=not caminho.exists(), index=False)


def imprimir_resumo (resultado: pd.DataFrame) -> None:
    todos = resultado[resultado["recorte"] == "todos"]
    tabela = todos.groupby("modelo", sort=False).agg(
        rmse_log=("rmse_log", "mean"), dp_rmse=("rmse_log", "std"),
        fator_mediano=("fator_mediano", "mean"), erro_rel_mediano=("erro_rel_mediano", "mean"),
        dentro_20pct=("dentro_20pct", "mean"), avaliacoes=("rmse_log", "size"),
    )
    print(tabela.round(4).to_string())
    if resultado["recorte"].str.startswith("cena:").any():
        por_cena = resultado[resultado["recorte"].str.startswith("cena:")]
        print("\nfator mediano por cena deixada de fora:")
        print(por_cena.pivot_table(index="recorte", columns="modelo", values="fator_mediano").round(3).to_string())


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alvo", choices=("vazao", "custo_fixo"), default="vazao")
    ap.add_argument("--esquema", choices=validacao.ESQUEMAS_CV, default="maquina")
    ap.add_argument("--modelos", default="B0,B1,B2,ridge,floresta,lgbm")
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--params", type=Path, default=None, help="JSON com hiperparâmetros do LightGBM")
    ap.add_argument("--teste", action="store_true", help="avalia UMA vez no teste blindado")
    ap.add_argument("--dataset", type=Path, default=None)
    args = ap.parse_args()

    df = dados.carregar(args.alvo, args.dataset)
    params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else None
    resultado = avaliar(df, args.alvo, args.modelos.split(","), args.esquema,
                        args.sementes, params, args.teste)
    registrar(resultado)
    imprimir_resumo(resultado)


if __name__ == "__main__":
    main()
