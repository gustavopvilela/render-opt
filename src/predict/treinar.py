from __future__ import annotations
import argparse
import json
import platform
import shutil
import time
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from comum.util import sha256_arquivo
from predict import dados, especificacoes, incerteza, metricas, modelos
from predict.ajustar import ler_params
from predict.comparar import metodo_escolhido

CSV_TESTE = dados.PASTA_EXPERIMENTOS / "teste_blindado.csv"


def teste_blindado (dev: dados.Conjunto, teste: dados.Conjunto, alvo: str) -> list[dict]:
    g = dev.grupos("maquina")
    linhas = []
    for metodo in ("B2",) + modelos.METODOS:
        m = modelos.criar(metodo, alvo, ler_params(metodo, alvo)).ajustar(dev.X, dev.y, g, dev.w)
        p = m.prever(teste.X)
        linhas.append({"alvo": alvo, "metodo": metodo, "recorte": "todos", **metricas.resumo(teste.y, p, teste.w)})
        for tipo in sorted(teste.df["dispositivo_tipo"].astype(str).unique()):
            s = (teste.df["dispositivo_tipo"].astype(str) == tipo).to_numpy()
            linhas.append({"alvo": alvo, "metodo": metodo, "recorte": f"tipo:{tipo}",
                           **metricas.resumo(teste.y[s], p[s], teste.w[s])})
    return linhas


def treinar_alvo (alvo: str, metodo: str, pasta: Path, caminho: Path | None = None,
                  medir_todos: bool = True, params: dict | None = None) -> tuple[dict, list[dict]]:
    params = ler_params(metodo, alvo) if params is None else params
    dev, teste = dados.separar(alvo, caminho)
    linhas = teste_blindado(dev, teste, alvo) if medir_todos else []

    g = dev.grupos("maquina")
    folga = incerteza.folga_maquina(dev, alvo, metodo, params)
    folga_cena = max(incerteza.folga_cena(incerteza.residuos_cena_nova(dev, alvo, metodo, params)), folga)
    m = modelos.criar(metodo, alvo, params).ajustar(dev.X, dev.y, g, dev.w)
    p = m.prever(teste.X)
    no_teste = metricas.resumo(teste.y, p, teste.w)
    cobertura = float(np.mean(incerteza.cobre(teste.y, incerteza.aplicar(p, folga, alvo), alvo)))

    todo = dados.conjunto(pd.concat([dev.df, teste.df], ignore_index=True), alvo)
    final = modelos.criar(metodo, alvo, params).ajustar(todo.X, todo.y, todo.grupos("maquina"), todo.w)
    joblib.dump(final, pasta / f"{alvo}.joblib", compress=3)

    print(f"{alvo:10s} {metodo}: teste rmse_log {no_teste['rmse_log']:.3f}, fator mediano {no_teste['fator_mediano']:.3f}; "
          f"folga {folga:.3f} (cobertura no teste {cobertura:.1%}), folga de cena nova {folga_cena:.3f}", flush=True)
    return {
        "metodo": metodo, "params": params or {}, "colunas": list(todo.cols), "categorias": todo.categorias,
        "folga_maquina_log": folga, "folga_cena_nova_log": folga_cena, "nivel": incerteza.NIVEL,
        "teste": no_teste, "cobertura_teste": cobertura, "linhas": len(todo.df),
        "linhas_por_fonte": todo.df["fonte"].value_counts().to_dict(),
        "maquinas": int(todo.df["maquina"].nunique()),
        "cenas": sorted(todo.df["cena_sha256"].astype(str).unique()),
        "maquinas_teste": int(teste.df["maquina"].nunique()),
    }, linhas


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metodo", choices=modelos.METODOS, default=None)
    ap.add_argument("--saida", type=Path, default=None)
    args = ap.parse_args()
    metodo = args.metodo or metodo_escolhido()

    pasta = args.saida or dados.RAIZ / "models" / time.strftime("%Y-%m-%dT%H-%M-%S")
    pasta.mkdir(parents=True, exist_ok=True)
    shutil.copy(especificacoes.SAIDA, pasta / "especificacoes.parquet")
    meta = {
        "criado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metodo": metodo,
        "datasets": {a: {"caminho": str(c), "sha256": sha256_arquivo(c)} for a, c in dados.DATASET.items()},
        "versoes": {"python": platform.python_version(), "lightgbm": lgb.__version__, "sklearn": sklearn.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "etapas_custo": ["carregar", "preparar_finalizar"],
        "alvos": {},
    }
    linhas, nomes = [], {}
    for alvo in dados.ALVOS:
        meta["alvos"][alvo], l = treinar_alvo(alvo, metodo, pasta)
        linhas += l
        nomes.update(dados.carregar(alvo).groupby("cena_sha256")["cena"].first().astype(str).to_dict())

    pd.DataFrame(linhas).to_csv(CSV_TESTE, index=False)
    (pasta / "metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (pasta / "cenas_treino.json").write_text(json.dumps(nomes, indent=2, ensure_ascii=False), encoding="utf-8")
    t = pd.DataFrame(linhas)
    print(t[t["recorte"] == "todos"].pivot_table(index="metodo", columns="alvo",
                                                 values=["rmse_log", "fator_mediano", "dentro_20pct"]).round(4).to_string())
    print(f"artefato: {pasta}\nteste blindado: {CSV_TESTE}")


if __name__ == "__main__":
    main()
