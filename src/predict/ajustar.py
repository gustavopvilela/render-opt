from __future__ import annotations
import argparse
import json
import numpy as np
import optuna
from predict import dados, metricas, modelos, validacao

PASTA_PARAMS = dados.PASTA_EXPERIMENTOS / "params"


def ler_params (metodo: str, alvo: str) -> dict | None:
    caminho = PASTA_PARAMS / f"{metodo}_{alvo}.json"
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def ajustar (metodo: str, alvo: str, tentativas: int) -> None:
    dev, _ = dados.separar(alvo)
    g = dev.grupos("maquina")
    divisao = validacao.dobras(g, "maquina", 5, semente=0)

    def objetivo (trial: optuna.Trial) -> float:
        params = modelos.espaco(metodo, trial)
        erros = []
        for k, (tr, va) in enumerate(divisao):
            m = modelos.criar(metodo, alvo, params).ajustar(dev.X.iloc[tr], dev.y[tr], g[tr], dev.w[tr])
            erros.append(metricas.resumo(dev.y[va], m.prever(dev.X.iloc[va]), dev.w[va])["rmse_log"])
            trial.report(float(np.mean(erros)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(erros))

    estudo = optuna.create_study(
        study_name=f"{metodo}_{alvo}", direction="minimize", load_if_exists=True,
        storage=f"sqlite:///{(dados.PASTA_EXPERIMENTOS / 'optuna.db').as_posix()}",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    estudo.enqueue_trial(modelos.PADROES[metodo], skip_if_exists=True)
    faltam = tentativas - len([t for t in estudo.trials if t.state.is_finished()])
    if faltam > 0:
        estudo.optimize(objetivo, n_trials=faltam)
    PASTA_PARAMS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_PARAMS / f"{metodo}_{alvo}.json"
    destino.write_text(json.dumps(estudo.best_params, indent=2), encoding="utf-8")
    padrao = estudo.trials[0].value
    print(f"{metodo:9s} {alvo:10s} padrão {padrao:.4f} -> melhor {estudo.best_value:.4f} "
          f"(tentativa {estudo.best_trial.number}) -> {destino.name}", flush=True)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metodos", default=",".join(modelos.METODOS))
    ap.add_argument("--alvos", default=",".join(dados.ALVOS))
    ap.add_argument("--tentativas", type=int, default=40)
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for alvo in args.alvos.split(","):
        for metodo in args.metodos.split(","):
            ajustar(metodo, alvo, args.tentativas)


if __name__ == "__main__":
    main()
