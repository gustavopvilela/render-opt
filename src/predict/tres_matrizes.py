"""
O experimento que decide se a IA vale a pena, rodado sem render farm.

    python -m predict.tres_matrizes --instancias 20 --maquinas 6 --frames 200
    python -m predict.tres_matrizes --cena-nova      # cada cena prevista sem tê-la visto

Cada instância sorteia máquinas do TESTE BLINDADO (nunca vistas pelos modelos) e
frames das 3 cenas do Open Data com quantidades de amostras variadas. Monta três matrizes:

  P_real   o "oráculo": medianas medidas no Open Data (c + S*W*H / v)
  P_prev   a previsão dos modelos
  P_cons   a previsão conservadora (vazão q10, custo q90)

e compara o makespan, sempre AVALIADO em P_real, de:

  (a) ordem de chegada                - sem modelo: a "Primitive Render Farm" do manual
                                        do Blender (e a política do Flamenco)
  (a') round-robin e blocos contíguos - sem modelo, divisões fixas da lista de frames
  (b) guloso sobre P_prev             - com o modelo
  (b') guloso sobre P_cons
  (c) guloso sobre P_real             - previsão perfeita
  LB  relaxação linear de R||Cmax     - ninguém faz melhor que isso

(a)->(b) é quanto a previsão vale; (b)->(c) é quanto ainda falta nela.
O custo fixo da simulação é só o 'preparar_finalizar', o único que o Open Data mede;
o 'carregar' fica de fora das três matrizes igualmente.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import wilcoxon
from scipy.sparse import coo_matrix, hstack
from master.planner import gerar_plano_renderizacao
from predict import dados, esquema, incerteza, modelos, validacao

# Pixels das cenas do Open Data
PIXELS = {"classroom": 1920 * 1080, "junkshop": 2000 * 1000, "monster": 1024 * 1024}


def makespan (plano: list[list[int]], P: np.ndarray) -> float:
    return max(float(P[seq, k].sum()) for k, seq in enumerate(plano) if seq) if any(plano) else 0.0


def round_robin (n: int, m: int) -> list[list[int]]:
    return [list(range(k, n, m)) for k in range(m)]


def contiguo (n: int, m: int) -> list[list[int]]:
    """Blocos de frames consecutivos do mesmo tamanho, como dividir a timeline entre as máquinas."""
    return [list(bloco) for bloco in np.array_split(np.arange(n), m)]


def ordem_de_chegada (P: np.ndarray) -> list[list[int]]:
    """
    A máquina que fica livre pega o próximo frame, na ordem da lista, sem olhar custo.
    P só diz quando cada máquina fica livre; empate (todas livres no início) vai para a de menor índice.
    """
    n, m = P.shape
    fim = np.zeros(m)
    plano: list[list[int]] = [[] for _ in range(m)]
    for i in range(n):
        k = int(np.argmin(fim))
        plano[k].append(i)
        fim[k] += P[i, k]
    return plano


def limite_inferior (P: np.ndarray) -> float:
    """Relaxação linear: min C, sum_j x_ij = 1, sum_i p_ij x_ij <= C, x >= 0."""
    n, m = P.shape
    linhas_i = np.repeat(np.arange(n), m)
    cols = np.arange(n * m)
    A_eq = hstack([coo_matrix((np.ones(n * m), (linhas_i, cols)), shape=(n, n * m)),
                   coo_matrix((n, 1))])
    A_ub = hstack([coo_matrix((P.ravel(), (np.tile(np.arange(m), n), cols)), shape=(m, n * m)),
                   coo_matrix(-np.ones((m, 1)))])
    c = np.zeros(n * m + 1)
    c[-1] = 1.0
    r = linprog(c, A_ub=A_ub.tocsr(), b_ub=np.zeros(m), A_eq=A_eq.tocsr(), b_eq=np.ones(n),
                bounds=(0, None), method="highs")
    return float(r.fun)


def prever (alvo: str, treino: pd.DataFrame, teste: pd.DataFrame, params: dict | None,
            cena_nova: bool) -> tuple[np.ndarray, np.ndarray]:
    """Previsão pontual e conservadora (com folga conformal) para as linhas de teste, em escala original."""
    cols = esquema.colunas(alvo)
    y = treino[esquema.COLUNA_ALVO[alvo]].to_numpy(float)
    w = treino["peso"].to_numpy(float)
    gm = validacao.grupos_de(treino, "maquina")
    X_tr, cats = esquema.montar_X(treino, cols)
    X_te, _ = esquema.montar_X(teste, cols, cats)
    ponto = modelos.criar("lgbm", alvo, params).ajustar(X_tr, y, gm, w)
    if cena_nova:
        folga = incerteza.folga_cena_nova(X_tr, y, gm, validacao.grupos_de(treino, "cena"), alvo, params, w)
        q = incerteza.ajustar_quantil(X_tr, y, gm, alvo, params, pesos=w)
    else:
        sub_tr, sub_cal = validacao.separar_teste(gm, 0.2, semente=1)
        q = incerteza.ajustar_quantil(X_tr.iloc[sub_tr], y[sub_tr], gm[sub_tr], alvo, params, pesos=w[sub_tr])
        folga = incerteza.calibrar(q.prever(X_tr.iloc[sub_cal]), y[sub_cal], alvo)
    return np.exp(ponto.prever(X_te)), np.exp(incerteza.aplicar(q.prever(X_te), folga, alvo))


def preparar (params_v: dict | None, params_c: dict | None, cena_nova: bool) -> pd.DataFrame:
    dv, dc = dados.carregar("vazao"), dados.carregar("custo_fixo")
    dev_v, te_v = validacao.teste_blindado(dv)
    dev_c, _ = validacao.teste_blindado(dc)
    tv, tc = dv.iloc[dev_v], dc.iloc[dev_c]
    # Simulação com o Open Data das máquinas do teste blindado
    t = dv.iloc[te_v]
    t = t[(t["fonte"] == "opendata") & t["log_custo_fixo"].notna()].assign(etapa="preparar_finalizar")
    if cena_nova:
        partes = [(tv[tv["cena_sha256"] != s], tc[tc["cena_sha256"] != s], t[t["cena_sha256"] == s])
                  for s in sorted(t["cena_sha256"].unique())]
    else:
        partes = [(tv, tc, t)]
    saida = []
    for treino_v, treino_c, alvo_t in partes:
        alvo_t = alvo_t.copy()
        alvo_t["v_prev"], alvo_t["v_cons"] = prever("vazao", treino_v, alvo_t, params_v, cena_nova)
        alvo_t["c_prev"], alvo_t["c_cons"] = prever("custo_fixo", treino_c, alvo_t, params_c, cena_nova)
        saida.append(alvo_t)
        print(f"previsto: {', '.join(sorted(alvo_t['cena'].unique()))}", flush=True)
    t = pd.concat(saida, ignore_index=True)
    # Uma linha por (máquina, versão, cena): nomes que só diferem antes da normalização viram uma
    chave = ["maquina", "versao_blender_major", "cena"]
    valores = ["custo_fixo_s", "vazao_amostras_pixel_s", "v_prev", "v_cons", "c_prev", "c_cons"]
    t = t.groupby(chave, as_index=False).agg({**{c: "median" for c in valores}, "dispositivo_tipo": "first"})
    # Só máquinas medidas nas 3 cenas
    completas = t.groupby(["maquina", "versao_blender_major"])["cena"].transform("nunique") == 3
    return t[completas]


def instancia (t: pd.DataFrame, rng: np.random.Generator, m: int, n: int) -> dict:
    chaves = t[["maquina", "versao_blender_major"]].drop_duplicates()
    escolha = chaves.sample(m, random_state=int(rng.integers(1 << 31)))
    linhas = t.merge(escolha, on=["maquina", "versao_blender_major"])
    cenas = rng.choice(sorted(linhas["cena"].unique()), size=n)
    amostras = np.exp(rng.uniform(np.log(64), np.log(2048), size=n)).round()
    trabalho = amostras * np.array([PIXELS[c] for c in cenas])

    def matriz (col_c: str, col_v: str) -> np.ndarray:
        P = np.empty((n, m))
        for k, (maq, ver) in enumerate(escolha.itertuples(index=False)):
            sub = linhas[(linhas["maquina"] == maq) & (linhas["versao_blender_major"] == ver)].set_index("cena")
            P[:, k] = sub.loc[cenas, col_c].to_numpy() + trabalho / sub.loc[cenas, col_v].to_numpy()
        return P

    # As três matrizes e os planos avaliados em P_real
    P_real = matriz("custo_fixo_s", "vazao_amostras_pixel_s")
    P_prev, P_cons = matriz("c_prev", "v_prev"), matriz("c_cons", "v_cons")
    planos = {
        "ordem_de_chegada": ordem_de_chegada(P_real), "round_robin": round_robin(n, m), "contiguo": contiguo(n, m),
        "modelo": gerar_plano_renderizacao(P_prev), "modelo_conservador": gerar_plano_renderizacao(P_cons),
        "oraculo": gerar_plano_renderizacao(P_real),
    }
    r = {f"cmax_{k}": makespan(p, P_real) for k, p in planos.items()}
    r["lb"] = limite_inferior(P_real)
    r["cmax_previsto_modelo"] = makespan(planos["modelo"], P_prev)
    r["tipos"] = ",".join(sorted(linhas.drop_duplicates("maquina")["dispositivo_tipo"]))
    return r


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instancias", type=int, default=20)
    ap.add_argument("--maquinas", type=int, default=6)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--cena-nova", action="store_true")
    ap.add_argument("--params-vazao", type=Path, default=None)
    ap.add_argument("--params-custo", type=Path, default=None)
    args = ap.parse_args()
    ler = lambda p: json.loads(p.read_text(encoding="utf-8")) if p else None

    # Previsões e instâncias
    t = preparar(ler(args.params_vazao), ler(args.params_custo), args.cena_nova)
    rng = np.random.default_rng(2026)
    r = pd.DataFrame([instancia(t, rng, args.maquinas, args.frames) for _ in range(args.instancias)])
    destino = dados.PASTA_EXPERIMENTOS / ("tres_matrizes_cena_nova.csv" if args.cena_nova else "tres_matrizes.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(destino, index=False)

    colunas = ["cmax_ordem_de_chegada", "cmax_round_robin", "cmax_contiguo", "cmax_modelo",
               "cmax_modelo_conservador", "cmax_oraculo"]
    razao = r[colunas].div(r["lb"], axis=0)
    print("makespan / limite inferior (média ± desvio entre instâncias; mediana):")
    for c in colunas:
        print(f"  {c[5:]:20s} {razao[c].mean():6.3f} ± {razao[c].std():.3f}   {razao[c].median():6.3f}")
    # A ordem de chegada é a referência principal: é o método nativo do Blender
    for base in ["ordem_de_chegada", "contiguo"]:
        ganho = 1 - r["cmax_modelo"] / r[f"cmax_{base}"]
        # Wilcoxon pareado
        teste = wilcoxon(r["cmax_modelo"], r[f"cmax_{base}"], alternative="less")
        print(f"redução do makespan, modelo vs {base}: {ganho.mean():.1%} ± {ganho.std():.1%}"
              f"  (Wilcoxon pareado, modelo < {base}: p = {teste.pvalue:.2g})")
    erro = (r["cmax_previsto_modelo"] - r["cmax_modelo"]) / r["cmax_modelo"]
    print(f"erro do makespan prometido pelo plano:  {erro.mean():+.1%} ± {erro.std():.1%}")
    print(f"escrito: {destino}")


if __name__ == "__main__":
    main()
