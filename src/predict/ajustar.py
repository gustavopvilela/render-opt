"""
Busca de hiperparâmetros do LightGBM com Optuna (TPE).

    python -m predict.ajustar --alvo vazao --tentativas 60

Usa só os dados FORA do teste blindado e a validação agrupada por máquina (a
pergunta principal). Grava os melhores parâmetros em
experiments/predict/params_<alvo>.json, que o avaliar.py e o treinar.py
aceitam com --params.
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import optuna
from predict import dados, esquema, metricas, modelos, validacao


def espaco (trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
    }


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alvo", choices=("vazao", "custo_fixo"), default="vazao")
    ap.add_argument("--tentativas", type=int, default=60)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    # Dados fora do teste blindado e dobras por máquina
    df = dados.carregar(args.alvo, args.dataset)
    dev, _ = validacao.teste_blindado(df)
    X, _ = esquema.montar_X(df.iloc[dev], esquema.colunas(args.alvo))
    y = df[esquema.COLUNA_ALVO[args.alvo]].to_numpy(float)[dev]
    w = df["peso"].to_numpy(float)[dev]
    g = validacao.grupos_de(df, "maquina")[dev]
    divisao = validacao.dobras(g, "maquina", 5, semente=0)

    # Objetivo: rmse médio em log nas dobras
    def objetivo (trial: optuna.Trial) -> float:
        params = espaco(trial)
        erros = []
        for k, (tr, va) in enumerate(divisao):
            m = modelos.criar("lgbm", args.alvo, params).ajustar(X.iloc[tr], y[tr], g[tr], w[tr])
            erros.append(metricas.resumo(y[va], m.prever(X.iloc[va]))["rmse_log"])
            trial.report(float(np.mean(erros)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(erros))

    dados.PASTA_EXPERIMENTOS.mkdir(parents=True, exist_ok=True)
    estudo = optuna.create_study(
        study_name=args.alvo, direction="minimize", load_if_exists=True,
        storage=f"sqlite:///{(dados.PASTA_EXPERIMENTOS / 'optuna.db').as_posix()}",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    # Estudo Optuna, começando pela configuração padrão
    estudo.enqueue_trial({"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20,
                          "colsample_bytree": 0.8, "subsample": 0.8, "reg_lambda": 1.0,
                          "reg_alpha": 1e-3, "min_split_gain": 0.0, "cat_smooth": 10.0},
                         skip_if_exists=True)
    estudo.optimize(objetivo, n_trials=args.tentativas)

    # Melhores parâmetros
    destino = dados.PASTA_EXPERIMENTOS / f"params_{args.alvo}.json"
    destino.write_text(json.dumps(estudo.best_params, indent=2), encoding="utf-8")
    print(f"melhor rmse_log: {estudo.best_value:.4f}  (tentativa {estudo.best_trial.number})")
    print(f"escrito: {destino}")


if __name__ == "__main__":
    main()
