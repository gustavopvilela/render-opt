from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from predict import dados, modelos, validacao

LADO = {"vazao": "inferior", "custo_fixo": "superior"}
NIVEL = 0.90


def escores (p: np.ndarray, y: np.ndarray, alvo: str) -> np.ndarray:
    return (y - p) if LADO[alvo] == "superior" else (p - y)


def quantil_conformal (s: np.ndarray, nivel: float = NIVEL) -> float:
    n = len(s)
    return float(np.quantile(s, min(np.ceil((n + 1) * nivel) / n, 1.0), method="higher"))


def quantil_ponderado (s: np.ndarray, w: np.ndarray, nivel: float = NIVEL) -> float:
    ordem = np.argsort(s)
    acumulado = np.cumsum(w[ordem]) / w.sum()
    return float(s[ordem][min(int(np.searchsorted(acumulado, nivel)), len(s) - 1)])


def aplicar (p: np.ndarray, folga: float, alvo: str) -> np.ndarray:
    return p + folga if LADO[alvo] == "superior" else p - folga


def cobre (y: np.ndarray, limite: np.ndarray, alvo: str) -> np.ndarray:
    return y <= limite if LADO[alvo] == "superior" else y >= limite


def folga_maquina (c: dados.Conjunto, alvo: str, metodo: str, params: dict | None, semente: int = 1) -> float:
    g = c.grupos("maquina")
    tr, cal = validacao.separar_teste(g, 0.2, semente)
    m = modelos.criar(metodo, alvo, params, semente).ajustar(c.X.iloc[tr], c.y[tr], g[tr], c.w[tr])
    return quantil_conformal(escores(m.prever(c.X.iloc[cal]), c.y[cal], alvo))


def residuos_cena_nova (c: dados.Conjunto, alvo: str, metodo: str, params: dict | None) -> pd.DataFrame:
    g, gc = c.grupos("maquina"), c.grupos("cena")
    partes = []
    for tr, va in validacao.dobras(gc, "cena"):
        m = modelos.criar(metodo, alvo, params).ajustar(c.X.iloc[tr], c.y[tr], g[tr], c.w[tr])
        partes.append(pd.DataFrame({"cena": gc[va], "escore": escores(m.prever(c.X.iloc[va]), c.y[va], alvo)}))
    return pd.concat(partes, ignore_index=True)


def folga_cena (residuos: pd.DataFrame, sem: str | None = None) -> float:
    r = residuos if sem is None else residuos[residuos["cena"] != sem]
    w = 1.0 / r.groupby("cena")["escore"].transform("size").to_numpy()
    return quantil_ponderado(r["escore"].to_numpy(), w)


def main () -> None:
    from predict.ajustar import ler_params
    from predict.comparar import metodo_escolhido
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alvo", choices=dados.ALVOS, default="vazao")
    ap.add_argument("--metodo", default=None)
    args = ap.parse_args()
    metodo = args.metodo or metodo_escolhido()
    params = ler_params(metodo, args.alvo)
    dev, _ = dados.separar(args.alvo)
    g = dev.grupos("maquina")

    linhas = []
    for k, (tr, va) in enumerate(validacao.dobras(g, "maquina", 5, semente=0)):
        folga = folga_maquina(dados.conjunto(dev.df.iloc[tr], args.alvo), args.alvo, metodo, params, semente=k)
        m = modelos.criar(metodo, args.alvo, params).ajustar(dev.X.iloc[tr], dev.y[tr], g[tr], dev.w[tr])
        p = m.prever(dev.X.iloc[va])
        linhas.append(pd.DataFrame({"tipo": dev.df["dispositivo_tipo"].iloc[va].to_numpy(), "folga": folga,
                                    "cru": cobre(dev.y[va], p, args.alvo),
                                    "conformal": cobre(dev.y[va], aplicar(p, folga, args.alvo), args.alvo)}))
    r = pd.concat(linhas)
    t = r.groupby("tipo")[["cru", "conformal"]].mean()
    t.loc["todos"] = r[["cru", "conformal"]].mean()
    print(f"[máquina nova] {metodo}, cobertura desejada {NIVEL:.0%} (limite {LADO[args.alvo]})")
    print(t.round(3).to_string())
    print(f"folga média (log): {r['folga'].mean():.4f}")

    res = residuos_cena_nova(dev, args.alvo, metodo, params)
    nomes = dev.df.groupby("cena_sha256")["cena"].first()
    print(f"\n[cena nova] cobertura na cena deixada de fora, folga calibrada nas outras")
    for cena, gr in res.groupby("cena"):
        f = folga_cena(res, sem=cena)
        print(f"  {nomes[cena]:14s} n={len(gr):5d}  sem folga {np.mean(gr['escore'] <= 0):.3f}  "
              f"folga {f:.3f} -> {np.mean(gr['escore'] <= f):.3f}")


if __name__ == "__main__":
    main()
