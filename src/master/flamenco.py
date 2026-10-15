"""
Simulação em C++ da distribuição do Flamenco (sa_core.flamenco). Nos testes reais, a mesma
regra roda no mestre, em services.jobs.entregar.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import sa_core
from master.planner import montar_instancia

ESPERA_INICIAL_S = 3.0 # state_awake.go do Flamenco
INTERVALO_ENTRE_TAREFAS_S = 5.0 # 2 s depois da tarefa + 3 s antes do pedido seguinte
TAMANHO_CHUNK_PADRAO = 1

@dataclass
class Simulacao:
    plano: list[list[int]]
    makespan: float
    maquina: np.ndarray
    inicio: np.ndarray # início do setup
    inicio_processamento: np.ndarray
    fim: np.ndarray
    tempos_maquinas: np.ndarray

def dividir_em_tarefas (frames_por_cena: Sequence[int], tamanho_chunk: int = TAMANHO_CHUNK_PADRAO) -> list[list[int]]:
    """O frameChunker do Flamenco, sem misturar cenas."""
    if tamanho_chunk < 1:
        raise ValueError("tamanho_chunk deve ser >= 1")
    tarefas, inicio = [], 0
    for n in frames_por_cena:
        tarefas += [list(range(i, min(i + tamanho_chunk, inicio + n))) for i in range(inicio, inicio + n, tamanho_chunk)]
        inicio += n
    return tarefas

def simular (
    instancia,
    *,
    tarefas: Optional[Sequence[Sequence[int]]] = None,
    espera_inicial: float = 0.0,
    intervalo_entre_tarefas: float = 0.0
) -> Simulacao:
    """
    :param tarefas: None = um job por tarefa, na ordem dos índices
    """
    plano, makespan, d = sa_core.flamenco(
        instancia,
        tarefas=None if tarefas is None else [list(t) for t in tarefas],
        espera_inicial=espera_inicial,
        intervalo_entre_tarefas=intervalo_entre_tarefas
    )
    return Simulacao(plano, makespan, d["maquina"], d["inicio"], d["inicio_processamento"], d["fim"], d["tempos_maquinas"])

def simular_farm (P: np.ndarray, C: Optional[np.ndarray] = None, **kwargs) -> Simulacao:
    return simular(montar_instancia(P, C), **kwargs)
