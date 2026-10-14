"""
Os modelos comparados, todos com a mesma interface:

    m = criar("lgbm", alvo="vazao")
    m.ajustar(X, y, grupos, pesos)   # y em log; pesos opcionais
    p = m.prever(X)                  # em log

Baselines primeiro: sem elas não dá para dizer se a IA vale alguma coisa.
"""

from __future__ import annotations
from typing import Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from predict.validacao import separar_teste, texto

# Baselines: mediana do grupo, com recuo para o grupo menor
_HW = ("dispositivo_tipo", "hw_familia", "hw_ano", "hw_classe")
TABELAS = {
    "vazao": {"B0": (), "B1": ("dispositivo_tipo",), "B2": _HW},
    "custo_fixo": {"B0": ("etapa",), "B1": ("etapa", "dispositivo_tipo"), "B2": ("etapa",) + _HW},
}

# LightGBM
PARAMS_LGBM = dict(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    max_cat_to_onehot=8,
    cat_smooth=10.0,
    deterministic=True, force_row_wise=True,
    verbose=-1,
)


class MedianaPorGrupo:
    """
    Tabela de consulta: mediana do alvo por combinação de colunas. Se a combinação
    não existe no treino, recua para as primeiras k-1 colunas, e assim por diante,
    até a mediana global. Sem colunas, é a mediana global (B0 da vazão).
    """

    def __init__ (self, cols: tuple[str, ...]):
        self.cols = list(cols)

    def ajustar (self, X, y, grupos=None, pesos=None):
        y = np.asarray(y)
        self.global_ = float(np.median(y))
        self.tabelas = []
        if self.cols:
            d = X[self.cols].apply(texto).assign(_y=y)
            self.tabelas = [(self.cols[:k], d.groupby(self.cols[:k])["_y"].median())
                            for k in range(len(self.cols), 0, -1)]
        return self

    def prever (self, X):
        if not self.cols:
            return np.full(len(X), self.global_)
        d = X[self.cols].apply(texto).reset_index(drop=True)
        p = pd.Series(np.nan, index=d.index)
        for cols, tabela in self.tabelas:
            falta = p.isna()
            if not falta.any():
                break
            chave = (pd.MultiIndex.from_frame(d.loc[falta, cols]) if len(cols) > 1
                     else pd.Index(d.loc[falta, cols[0]]))
            p.loc[falta] = tabela.reindex(chave).to_numpy()
        return p.fillna(self.global_).to_numpy()


class Sklearn:
    """Adapta um pipeline do scikit-learn à interface comum."""

    def __init__ (self, pipeline):
        self.pipeline = pipeline

    def ajustar (self, X, y, grupos=None, pesos=None):
        ultimo = self.pipeline.steps[-1][0]
        extra = {} if pesos is None else {f"{ultimo}__sample_weight": np.asarray(pesos)}
        self.pipeline.fit(X, y, **extra)
        return self

    def prever (self, X):
        return self.pipeline.predict(X)


def _ridge () -> Sklearn:
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()),
         make_column_selector(dtype_exclude="category")),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include="category")),
    ])
    return Sklearn(make_pipeline(prep, RidgeCV(alphas=np.logspace(-3, 3, 13))))


def _floresta (semente: int) -> Sklearn:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), make_column_selector(dtype_exclude="category")),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), make_column_selector(dtype_include="category")),
    ])
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=semente)
    return Sklearn(make_pipeline(prep, rf))


class LGBM:
    """
    LightGBM com parada antecipada. O conjunto que decide a parada sai dos GRUPOS
    do próprio treino, nunca do teste: usar o teste aqui é o vazamento mais comum.
    """

    def __init__ (self, params: Optional[dict] = None, semente: int = 0, fracao_parada: float = 0.15):
        self.params = {**PARAMS_LGBM, **(params or {}), "random_state": semente}
        self.semente = semente
        self.fracao_parada = fracao_parada

    def ajustar (self, X, y, grupos, pesos=None):
        y = np.asarray(y)
        w = None if pesos is None else np.asarray(pesos, float)
        tr, va = separar_teste(np.asarray(grupos), self.fracao_parada, self.semente)
        self.modelo = lgb.LGBMRegressor(**self.params)
        self.modelo.fit(X.iloc[tr], y[tr], sample_weight=None if w is None else w[tr],
                        eval_X=(X.iloc[va],), eval_y=(y[va],),
                        callbacks=[lgb.early_stopping(200, verbose=False)])
        self.melhor_iteracao = int(self.modelo.best_iteration_ or self.params["n_estimators"])
        return self

    def prever (self, X):
        return self.modelo.predict(X, num_iteration=self.melhor_iteracao)


def criar (nome: str, alvo: str, params: Optional[dict] = None, semente: int = 0):
    if nome in ("B0", "B1", "B2"):
        return MedianaPorGrupo(TABELAS[alvo][nome])
    if nome == "ridge":
        return _ridge()
    if nome == "floresta":
        return _floresta(semente)
    if nome == "lgbm":
        return LGBM(params, semente)
    raise ValueError(f"Modelo desconhecido: {nome}")
