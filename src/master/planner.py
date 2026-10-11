"""
Onde o mestre toma decisões como estimar tempos, monstar o plano, decidir
encerramentos e redistribuir os jobs.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from comum.protocolo import DecisaoExcesso, Progresso
from comum.util import versao_tupla
from config import ConfigMestre

@dataclass
class Maquina:
    id: str
    hardware: dict
    versoes_blender: list[str]
    dispositivo: str

@dataclass
class Cena:
    id: str
    caminho: str
    versao: str

def estimar_matriz_p (cena: Cena, frames: Sequence[int], maquinas: Sequence[Maquina]) -> np.ndarray:
    elegivel = matriz_elegibilidade(cena, maquinas)

    # TODO: AQUI VEM A IA, O CÓDIGO A SEGUIR É PROVISÓRIO
    P = np.empty((len(frames), len(maquinas)))
    for k, maquina in enumerate(maquinas):
        P[:, k] = 60.0 / float(maquina.hardware.get("fator_velocidade", 1.0))

    P[:, ~elegivel] = math.inf
    return P

def matriz_elegibilidade (cena: Cena, maquinas: Sequence[Maquina]) -> np.ndarray:
    """
    Retorna True quando a máquina puder renderizar a cena.
    O mínimo é ter uma versão do Blender igual ou maior à da cena.
    """
    versao = versao_tupla(cena.versao)
    return np.array([
        any(versao_tupla(v) >= versao for v in maquina.versoes_blender)
        for maquina in maquinas
    ], dtype=bool)

def gerar_plano_renderizacao (P: np.ndarray, semente: Optional[int] = None) -> list[list[int]]:
    n, m = P.shape
    if n == 0:
        return [[] for _ in range(m)]
    if np.isinf(P).all(axis=1).any():
        i = int(np.flatnonzero(np.isinf(P).all(axis=1))[0])
        raise ValueError(f"O frame da linha {i} não tem nenhuma máquina elegível")

    # TODO: AQUI ENTRA A METAHEURÍSTICA, O CÓDIGO A SEGUIR É PROVISÓRIO
    custo_minimo = np.where(np.isinf(P), np.nan, P)
    ordem = np.argsort(-np.nanmin(custo_minimo, axis=1))
    fim = np.zeros(m)
    plano: list[list[int]] = [[] for _ in range(m)]
    for i in ordem:
        k = int(np.argmin(fim + P[i]))
        plano[k].append(int(i))
        fim += P[i, k]
    return plano

def projetar_total (decorrido: float, progresso: Progresso) -> Optional[float]:
    """
    Tempo total projetado do job a partir do progresso, ou None sem progresso.
    """
    atual, total = progresso.amostra_atual, progresso.amostras_total
    if not atual or not total:
        return None
    preparo = progresso.tempo_preparacao_s or 0.0
    instante = progresso.instante_amostra_s if progresso.instante_amostra_s is not None else decorrido
    projetado = preparo + (instante - preparo) * total / atual
    return max(projetado, decorrido)

def decidir_excesso (
    *,
    previsto_aqui: float,
    decorrido: float,
    progresso: Progresso,
    previstos_outras: dict[str, float],
    config: ConfigMestre
) -> DecisaoExcesso:
    """
    Se o job passou de 1,5x o tempo previsto, toma-se uma decisão com base em duas políticas:

    Política simples: para o processo.

    Política de comparação (padrão): para o processo na máquina atual do job somente se nesta
    demorar mais para terminar do que colocar em outra máquina desde o início.
    """
    if config.politica_excesso == "simples":
        return DecisaoExcesso(
            acao="abortar",
            motivo="Excedeu o limite (política simples)"
        )

    escala_teto = config.fator_teto / config.fator_excesso
    if not previstos_outras: # Não há outra máquina para a qual mandar o job
        return DecisaoExcesso(
            acao="continuar",
            motivo="Nenhuma outra máquina elegível",
            novo_teto_s=decorrido * escala_teto,
        )

    total_projetado = projetar_total(decorrido, progresso)
    if total_projetado is None: # Sem progresso legível
        return DecisaoExcesso(
            acao="abortar",
            motivo="Excedeu o limite e não há progresso para projetar"
        )

    falta_aqui = max(total_projetado - decorrido, 0.0)
    # Se o modelo errou nesta cena, é provável que errou em outras máquinas também,
    # portanto, corrige-se as previsões delas pelo mesmo fator
    fator_erro = max(total_projetado / previsto_aqui, 1.0)
    melhor_outra = min(previstos_outras.values(), default=math.inf) * fator_erro

    if falta_aqui > config.vantagem_minima * melhor_outra:
        return DecisaoExcesso(
            acao="abortar",
            motivo=f"Faltam ~{falta_aqui:.2f} s aqui, refazer em outra máquina leva ~{melhor_outra:.2f} s"
        )

    # Confere novamente se a projeção estoura em 25%, se sim, o teto sobe n mesma proporção
    proximo = max(total_projetado * 1.25, decorrido + config.folga_minima_s)
    return DecisaoExcesso(
        acao="continuar",
        motivo=f"Faltam ~{falta_aqui:.2f} s, refazer em outra máquina levaria ~{melhor_outra:.2f} s",
        proximo_checkpoint_s=proximo,
        novo_teto_s=proximo * escala_teto
    )

def escolher_destino(
    estimativas: dict[str, float],
    cargas: dict[str, float],
    excluir: Optional[str] = None
) -> Optional[str]:
    """
    Decide para onde vai um job que voltou para a fila (abortado, falhou, ou era de
    uma máquina que caiu)
    :param estimativas: processamento do job em cada máquina elegível
    :param cargas: segundos de trabalho já na fila de cada máquina elegível
    :param excluir: máquina de onde original saiu o job
    :return: nova máquina de destino do job
    """
    # TODO: AQUI PODE ENTRAR A METAHEURÍSTICA PARA REALOCAR TODOS OS JOBS PENDENTES
    candidatas = {m: estimativas[m] + cargas[m] for m in cargas if m in estimativas}
    if excluir is not None and len(candidatas) > 1:
        candidatas.pop(excluir, None)
    if not candidatas:
        return None
    return min(candidatas, key=candidatas.get)

def escolher_roubo (
    ocioso: str,
    filas: dict[str, list[str]],
    estimativas: dict[str, dict[str, float]],
    cargas: dict[str, float]
) -> Optional[str]:
    """
    Acontece quando a fila de uma máquina acabou mas ainda há jobs em
    outras máquinas.
    :param ocioso: máquina ociosa
    :param filas: filas de jobs de cada máquina
    :param estimativas: processamento previsto de cada job
    :param cargas: segundos de trabalho na fila
    :return: id de um job a ser transferido para a máquina ociosa, ou None se não compensar.
    """
    # TODO: AQUI PODE ENTRAR A METAHEURÍSTICA PARA REALOCAR OS JOBS
    melhor, maior_ganho = None, 0.0
    for dona, jobs in filas.items():
        if dona == ocioso or not jobs:
            continue
        job = jobs[-1]
        aqui = estimativas[job].get(ocioso)
        if aqui is None:
            continue
        termino_na_dona = cargas[dona]
        ganho = termino_na_dona - aqui
        if ganho > maior_ganho:
            melhor, maior_ganho = job, ganho
    return melhor