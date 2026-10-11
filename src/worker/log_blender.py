"""
Leitura do log do Blender enquando ele renderiza.

00:03.563 render | Fra: 1 | Mem: 284M | Sample 0/8
00:03.672 render | Fra: 1 | Remaining: 00:00.70 | Mem: 288M | Sample 1/8
00:04.219 render | Fra: 1 | Mem: 288M | Sample 8/8
00:04.344 render | Saved: 'C:\\...\\out_0001.png'
00:04.344 render | Time: 00:01.94 (Saving: 00:00.02)

Na serie 4.x o comeco da linha muda ("Fra:1 Mem:24.08M (Peak 25.43M) | ...
| Sample 1/128"), mas os trechos que importam sao os mesmos. Se uma versao
nova mudar o formato, somente este arquivo precisa mudar.
"""

from __future__ import annotations
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

RE_AMOSTRA = re.compile(r"Sample (\d+)/(\d+)")
RE_MEMORIA = re.compile(r"Mem: ?([\d.]+)M")
RE_SALVO = re.compile(r"Saved: '(.+)'")
RE_TEMPO_FINAL = re.compile(r"Time: ((?:\d+:)?\d+:\d+(?:\.\d+)?) \(Saving: ((?:\d+:)?\d+:\d+(?:\.\d+)?)\)")
RE_VERSAO = re.compile(r"^Blender (\d+\.\d+)")

def segundos (texto: str) -> float:
    """
    Transforma o timestamp do Blender em segundos. Ex.: '01:02.50' -> 62.5
    """
    total = 0.0
    for parte in texto.split(":"):
        total = total * 60 + float(parte)
    return total

@dataclass
class AnaliseLog:
    tempo_preparacao_s: Optional[float] = None    # Instante da primeira linha "Sample x/N"
    tempo_salvo_s: Optional[float] = None       # Instante da linha "Saved:"
    amostra_atual: Optional[int] = None
    amostras_total: Optional[int] = None
    instante_amostra_s: Optional[float] = None  # Quando a última linha "Sample x/N" chegou
    memoria_max_mb: Optional[float] = None
    tempo_blender_s: Optional[float] = None
    tempo_gravacao_s: Optional[float] = None
    versao_blender: Optional[str] = None
    ultimas_linhas: deque = field(default_factory=lambda: deque(maxlen=40))

    def alimentar (self, linha: str, instante: float) -> None:
        """
        :param instante: segundos desde o lançamento do Blender
        """
        self.ultimas_linhas.append(linha)

        if m := RE_AMOSTRA.search(linha):
            if self.tempo_preparacao_s is None:
                self.tempo_preparacao_s = instante
            self.amostra_atual, self.amostras_total = int(m[1]), int(m[2])
            self.instante_amostra_s = instante
        if m := RE_MEMORIA.search(linha):
            mb = float(m[1])
            self.memoria_max_mb = max(mb, self.memoria_max_mb or 0.0)
        if RE_SALVO.search(linha):
            self.tempo_salvo_s = instante
        if m := RE_TEMPO_FINAL.search(linha):
            self.tempo_blender_s, self.tempo_gravacao_s = segundos(m[1]), segundos(m[2])
        if self.versao_blender is None and (m := RE_VERSAO.match(linha)):
            self.versao_blender = m[1]

    @property
    def tempo_amostragem_s (self) -> Optional[float]:
        if self.tempo_preparacao_s is None or self.tempo_salvo_s is None:
            return None
        return self.tempo_salvo_s - self.tempo_preparacao_s