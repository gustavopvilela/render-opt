"""
Partições de treino e teste.

A unidade de partição é o GRUPO, nunca a linha: linhas do mesmo grupo
(a mesma máquina, ou os frames da mesma cena) são quase cópias umas das outras.
Se caírem dos dois lados, o teste mede memória, não generalização.
"""

from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, LeaveOneGroupOut

ESQUEMAS_CV = ("maquina", "cena", "familia", "aleatorio")


def texto (s: pd.Series) -> pd.Series:
    """Converte para texto SEM perder os ausentes: NaN vira "nan", e não some do grupo."""
    return s.astype(object).map(str)


def grupos_de (df: pd.DataFrame, esquema: str) -> np.ndarray:
    """
    maquina:   generaliza para uma máquina nunca vista? (a pergunta principal)
    cena:      generaliza para uma cena nunca vista? (RN-12; grupo = SHA-256 do .blend)
    familia:   generaliza para uma GERAÇÃO de hardware nunca vista?
    aleatorio: nenhuma pergunta; mostra quanto a validação errada infla o resultado
    """
    if esquema == "maquina":
        return texto(df["maquina"]).to_numpy()
    if esquema == "cena":
        return texto(df["cena_sha256"]).to_numpy()
    if esquema == "familia":
        return (texto(df["dispositivo_tipo"]) + "|" + texto(df["hw_familia"]) + "|"
                + texto(df["hw_ano"])).to_numpy()
    if esquema == "aleatorio":
        return np.arange(len(df)).astype(str)
    raise ValueError(f"Esquema desconhecido: {esquema}")


def teste_blindado (df: pd.DataFrame, fracao: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Teste blindado: as máquinas cujo SHA-256 do nome cai abaixo de 'fracao'.
    Não depende de quais outras máquinas estão no dataset: a mesma máquina fica sempre
    do mesmo lado, nos dois datasets (vazão e custo) e depois de cada retreino.
    :return: (posições do desenvolvimento, posições do teste)
    """
    h = df["maquina"].astype(str).map(lambda m: int(hashlib.sha256(m.encode("utf-8")).hexdigest()[:8], 16) / 16 ** 8)
    teste = (h < fracao).to_numpy()
    return np.flatnonzero(~teste), np.flatnonzero(teste)


def separar_teste (grupos: np.ndarray, fracao: float = 0.2, semente: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Uma fração dos GRUPOS para um lado (parada antecipada, calibração). Sorteio com semente."""
    gss = GroupShuffleSplit(n_splits=1, test_size=fracao, random_state=semente)
    treino, teste = next(gss.split(np.zeros(len(grupos)), groups=grupos))
    return treino, teste


def dobras (grupos: np.ndarray, esquema: str, n_dobras: int = 5, semente: int = 0) -> list:
    """
    Índices (treino, validação) de cada dobra.
    - cena: deixa uma cena de fora por vez (há poucas cenas; a semente não muda nada);
    - aleatorio: dobras por linha;
    - os outros: GroupKFold embaralhado; sementes diferentes distribuem os grupos de outro jeito.
    """
    x = np.zeros(len(grupos))
    if esquema == "cena":
        return list(LeaveOneGroupOut().split(x, groups=grupos))
    if esquema == "aleatorio":
        return list(KFold(n_dobras, shuffle=True, random_state=semente).split(x))
    n = min(n_dobras, len(np.unique(grupos)))
    return list(GroupKFold(n, shuffle=True, random_state=semente).split(x, groups=grupos))
