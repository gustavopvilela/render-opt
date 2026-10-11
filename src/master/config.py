from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
import yaml
from pydantic import BaseModel

"""
A configuração é lida de um arquivo YAML
"""
class ConfigMestre (BaseModel):
    host: str = "0.0.0.0"
    port: int = 9678
    token: Optional[str] = None
    pasta_dados: Path
    blender: dict[str, str] = {} # Versão -> Blender do mestre usado para empacotar as cenas

    intervalo_heartbeat_s: float = 10
    intervalo_ocioso_s: float = 5
    espera_longa_s: float = 25
    limite_silencio_s: float = 60 # Sem heartbeat por mais que isso, significa que o trabalhador caiu
    max_tentativas: int = 3

    politica_excesso: Literal["simples", "comparar"] = "comparar"
    fator_excesso: float = 1.5
    folga_minima_s: float = 60
    vantagem_minima: float = 1.25 # Só aborta se outra máquina for 25% melhor
    max_adiamentos: int = 1 # Depois disso, o job roda até o final
    fator_teto: float = 4.0
    limite_sem_saida_s: float = 600 # Se o Blender ficar calado por mais de 10 minutos, considera como travado

    @property
    def caminho_db (self) -> Path:
        return self.pasta_dados / "estado.sqlite3"

    @property
    def pasta_cenas (self) -> Path:
        return self.pasta_dados / "cenas"

    @property
    def pasta_resultados (self) -> Path:
        return self.pasta_dados / "resultados"

def carregar_config (caminho: Path) -> ConfigMestre:
    with caminho.open(encoding="utf-8") as arquivo:
        return ConfigMestre.model_validate(yaml.safe_load(arquivo))
