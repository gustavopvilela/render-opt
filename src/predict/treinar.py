"""
Treina os modelos finais e grava o artefato que o mestre carrega.

    python -m predict.treinar --params-vazao experiments/predict/params_vazao.json \
                              --params-custo experiments/predict/params_custo_fixo.json

Para cada alvo (vazão e custo fixo):
  - treina nas máquinas de desenvolvimento e mede UMA vez no teste blindado;
  - treina o quantil (q10 da vazão, q90 do custo) e calibra as duas folgas conformais:
    a de máquina nova (cena conhecida) e a de cena nova (deixa uma cena de fora);
  - retreina tudo com TODOS os dados, que é o que vai para produção.

Saída: models/<data-hora>/, com um arquivo por modelo no formato texto do LightGBM
(estável entre versões, ao contrário do pickle), os metadados e a lista de cenas do treino.
"""

from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from comum.util import sha256_arquivo
from predict import dados, esquema, incerteza, metricas, modelos, validacao


def treinar_alvo (df: pd.DataFrame, alvo: str, params: dict | None, pasta: Path) -> dict:
    cols = esquema.colunas(alvo)
    y = df[esquema.COLUNA_ALVO[alvo]].to_numpy(float)
    w = df["peso"].to_numpy(float)
    gm, gc = validacao.grupos_de(df, "maquina"), validacao.grupos_de(df, "cena")
    dev, teste = validacao.teste_blindado(df)
    X_dev, cats = esquema.montar_X(df.iloc[dev], cols)
    X_te, _ = esquema.montar_X(df.iloc[teste], cols, cats)

    # Desempenho no teste blindado
    m = modelos.criar("lgbm", alvo, params).ajustar(X_dev, y[dev], gm[dev], w[dev])
    no_teste = metricas.resumo(y[teste], m.prever(X_te))

    # Quantil e folgas conformais
    sub_tr, sub_cal = validacao.separar_teste(gm[dev], 0.2, semente=1)
    q = incerteza.ajustar_quantil(X_dev.iloc[sub_tr], y[dev][sub_tr], gm[dev][sub_tr], alvo, params, pesos=w[dev][sub_tr])
    folga = incerteza.calibrar(q.prever(X_dev.iloc[sub_cal]), y[dev][sub_cal], alvo)
    cobertura = float(np.mean(incerteza.cobre(y[teste], incerteza.aplicar(q.prever(X_te), folga, alvo), alvo)))
    folga_cena = incerteza.folga_cena_nova(X_dev, y[dev], gm[dev], gc[dev], alvo, params, w[dev])

    # Modelos de produção, com todos os dados
    X_todo, cats = esquema.montar_X(df, cols)
    final = modelos.criar("lgbm", alvo, params).ajustar(X_todo, y, gm, w)
    final_q = incerteza.ajustar_quantil(X_todo, y, gm, alvo, params, pesos=w)
    final.modelo.booster_.save_model(pasta / f"{alvo}_mediana.txt", num_iteration=final.melhor_iteracao)
    final_q.modelo.booster_.save_model(pasta / f"{alvo}_quantil.txt", num_iteration=final_q.melhor_iteracao)

    print(f"{alvo:10s} teste: fator mediano {no_teste['fator_mediano']:.3f}, rmse_log {no_teste['rmse_log']:.3f}; "
          f"cobertura do quantil {cobertura:.1%}; folga {folga:.3f}, folga de cena nova {folga_cena:.3f}")
    return {
        "colunas": list(cols), "categorias": cats,
        "folga_conformal_log": folga, "folga_cena_nova_log": max(folga_cena, folga),
        "quantil": incerteza.QUANTIL[alvo][0], "teste": no_teste, "cobertura_teste": cobertura,
        "iteracoes": final.melhor_iteracao, "linhas": len(df),
        "linhas_por_fonte": df["fonte"].value_counts().to_dict(),
        "maquinas": int(df["maquina"].nunique()),
        "cenas": sorted(df["cena_sha256"].astype(str).unique()),
        "maquinas_teste": sorted(df["maquina"].iloc[teste].astype(str).unique()),
    }


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params-vazao", type=Path, default=None)
    ap.add_argument("--params-custo", type=Path, default=None)
    ap.add_argument("--saida", type=Path, default=None)
    args = ap.parse_args()

    pasta = args.saida or dados.RAIZ / "models" / time.strftime("%Y-%m-%dT%H-%M-%S")
    pasta.mkdir(parents=True, exist_ok=True)
    ler = lambda p: json.loads(p.read_text(encoding="utf-8")) if p else None
    meta = {
        "criado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": {alvo: {"caminho": str(c), "sha256": sha256_arquivo(c)} for alvo, c in dados.DATASET.items()},
        "versoes": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                    "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "etapas_custo": list(esquema.ETAPAS),
        "alvos": {},
    }
    nomes = {}
    # Um modelo por alvo
    for alvo, params in (("vazao", ler(args.params_vazao)), ("custo_fixo", ler(args.params_custo))):
        df = dados.carregar(alvo)
        meta["alvos"][alvo] = treinar_alvo(df, alvo, params, pasta) | {"params": params or {}}
        nomes.update(df.groupby("cena_sha256")["cena"].first().astype(str).to_dict())

    (pasta / "metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (pasta / "cenas_treino.json").write_text(json.dumps(nomes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"artefato: {pasta}")


if __name__ == "__main__":
    main()
