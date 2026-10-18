from __future__ import annotations
import copy
from typing import Optional
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, QuantileTransformer
from predict.validacao import separar_teste, texto

METODOS = ("lgbm", "floresta", "rede")
NOMES = {"lgbm": "LightGBM", "floresta": "Random Forest", "rede": "Rede neural (MLP)", "B2": "Mediana por hardware"}

_HW = ("dispositivo_tipo", "hw_familia", "hw_ano", "hw_classe")
TABELAS = {"vazao": _HW, "custo_fixo": ("etapa",) + _HW}


class MedianaPorGrupo:
    def __init__ (self, cols: tuple[str, ...]):
        self.cols = list(cols)

    def ajustar (self, X, y, grupos=None, pesos=None):
        y = np.asarray(y)
        self.global_ = float(np.median(y))
        d = X[self.cols].apply(texto).assign(_y=y)
        self.tabelas = [(self.cols[:k], d.groupby(self.cols[:k])["_y"].median())
                        for k in range(len(self.cols), 0, -1)]
        return self

    def prever (self, X):
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


PADRAO_LGBM = dict(
    learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=1e-3, min_split_gain=0.0, cat_smooth=10.0,
)


class LGBM:
    def __init__ (self, params: Optional[dict] = None, semente: int = 0, fracao_parada: float = 0.15):
        self.params = {**PADRAO_LGBM, **(params or {})}
        self.semente = semente
        self.fracao_parada = fracao_parada

    def ajustar (self, X, y, grupos, pesos=None):
        y = np.asarray(y)
        w = None if pesos is None else np.asarray(pesos, float)
        tr, va = separar_teste(np.asarray(grupos), self.fracao_parada, self.semente)
        self.modelo = lgb.LGBMRegressor(
            objective="regression", n_estimators=5000, subsample_freq=1, max_cat_to_onehot=8,
            deterministic=True, force_row_wise=True, verbose=-1, random_state=self.semente, **self.params)
        self.modelo.fit(X.iloc[tr], y[tr], sample_weight=None if w is None else w[tr],
                        eval_X=(X.iloc[va],), eval_y=(y[va],),
                        eval_sample_weight=None if w is None else [w[va]],
                        callbacks=[lgb.early_stopping(200, verbose=False)])
        self.iteracoes = int(self.modelo.best_iteration_ or 5000)
        return self

    def prever (self, X):
        return self.modelo.predict(X, num_iteration=self.iteracoes)

    @staticmethod
    def espaco (trial: optuna.Trial) -> dict:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
            "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        }


PADRAO_FLORESTA = dict(max_features=1 / 3, min_samples_leaf=3, max_samples=1.0)
ARVORES = 300


def _categorias_como_texto (X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(object).where(X.notna(), "ausente").astype(str)


class Floresta:
    def __init__ (self, params: Optional[dict] = None, semente: int = 0):
        self.params = {**PADRAO_FLORESTA, **(params or {})}
        self.semente = semente

    def ajustar (self, X, y, grupos=None, pesos=None):
        prep = ColumnTransformer([
            ("cat", make_pipeline(FunctionTransformer(_categorias_como_texto),
                                  OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
             make_column_selector(dtype_include="category")),
        ], remainder="passthrough")
        rf = RandomForestRegressor(n_estimators=ARVORES, n_jobs=-1, random_state=self.semente, **self.params)
        self.modelo = make_pipeline(prep, rf)
        self.modelo.fit(X, np.asarray(y), randomforestregressor__sample_weight=None if pesos is None else np.asarray(pesos, float))
        return self

    def prever (self, X):
        return self.modelo.predict(X)

    @staticmethod
    def espaco (trial: optuna.Trial) -> dict:
        return {
            "max_features": trial.suggest_float("max_features", 0.05, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40, log=True),
            "max_samples": trial.suggest_float("max_samples", 0.3, 1.0),
        }


PADRAO_REDE = dict(camadas=2, largura=256, alpha=1e-4, learning_rate_init=1e-3, batch_size=256)
EPOCAS = 300
PACIENCIA = 16


def preprocessamento_rede (n_linhas: int) -> ColumnTransformer:
    numericas = make_column_selector(dtype_exclude="category")
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, n_linhas),
                                                  subsample=None, random_state=0)), numericas),
        ("ausente", MissingIndicator(features="missing-only", error_on_new=False), numericas),
        ("cat", make_pipeline(FunctionTransformer(_categorias_como_texto),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                            sparse_output=False)),
         make_column_selector(dtype_include="category")),
    ])


class Rede:
    def __init__ (self, params: Optional[dict] = None, semente: int = 0, fracao_parada: float = 0.15):
        self.params = {**PADRAO_REDE, **(params or {})}
        self.semente = semente
        self.fracao_parada = fracao_parada

    def ajustar (self, X, y, grupos, pesos=None):
        y = np.asarray(y, float)
        w = np.ones(len(y)) if pesos is None else np.asarray(pesos, float)
        tr, va = separar_teste(np.asarray(grupos), self.fracao_parada, self.semente)
        self.prep = preprocessamento_rede(len(tr)).fit(X.iloc[tr])
        Z_tr, Z_va = self.prep.transform(X.iloc[tr]), self.prep.transform(X.iloc[va])
        self.media, self.desvio = float(np.average(y[tr], weights=w[tr])), float(np.std(y[tr])) or 1.0
        alvo_tr, alvo_va = (y[tr] - self.media) / self.desvio, (y[va] - self.media) / self.desvio
        p = self.params
        self.rede = MLPRegressor(hidden_layer_sizes=(int(p["largura"]),) * int(p["camadas"]), activation="relu",
                                 solver="adam", alpha=p["alpha"], learning_rate_init=p["learning_rate_init"],
                                 batch_size=int(p["batch_size"]), shuffle=True, random_state=self.semente)
        melhor, sem_melhora, self.epocas = np.inf, 0, 0
        for epoca in range(EPOCAS):
            self.rede.partial_fit(Z_tr, alvo_tr, sample_weight=w[tr])
            erro = float(np.average((self.rede.predict(Z_va) - alvo_va) ** 2, weights=w[va]))
            if erro < melhor - 1e-6:
                melhor, sem_melhora, self.epocas = erro, 0, epoca + 1
                guardado = (copy.deepcopy(self.rede.coefs_), copy.deepcopy(self.rede.intercepts_))
            else:
                sem_melhora += 1
                if sem_melhora >= PACIENCIA:
                    break
        self.rede.coefs_, self.rede.intercepts_ = guardado
        return self

    def prever (self, X):
        return self.rede.predict(self.prep.transform(X)) * self.desvio + self.media

    @staticmethod
    def espaco (trial: optuna.Trial) -> dict:
        return {
            "camadas": trial.suggest_int("camadas", 1, 4),
            "largura": trial.suggest_int("largura", 64, 512, log=True),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }


CLASSES = {"lgbm": LGBM, "floresta": Floresta, "rede": Rede}
PADROES = {"lgbm": PADRAO_LGBM, "floresta": PADRAO_FLORESTA, "rede": PADRAO_REDE}


def criar (nome: str, alvo: str, params: Optional[dict] = None, semente: int = 0):
    if nome == "B2":
        return MedianaPorGrupo(TABELAS[alvo])
    if nome not in CLASSES:
        raise ValueError(f"Método desconhecido: {nome}")
    return CLASSES[nome](params, semente)


def espaco (nome: str, trial: optuna.Trial) -> dict:
    return CLASSES[nome].espaco(trial)
