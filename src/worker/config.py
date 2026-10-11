"""
Configuração do trabalhador, vinda do arquivo YAML.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import yaml
from pydantic import BaseModel
from comum.protocolo import Id

class ConfigTrabalhador (BaseModel):
    mestre: str
    trabalhador_id: Id
    token: Optional[str] = None
    dispositivo: str = "CPU"
    pasta_trabalho: Path
    blender: dict[str, Union[str, list[str]]]
    fator_velocidade: Optional[float] = None

    @property
    def pasta_cenas (self) -> Path:
        return self.pasta_trabalho / "cenas"

    @property
    def pasta_jobs (self) -> Path:
        return self.pasta_trabalho / "jobs"

def carregar_config (caminho: Path) -> ConfigTrabalhador:
    with caminho.open(encoding="utf-8") as arquivo:
        return ConfigTrabalhador.model_validate(yaml.safe_load(arquivo))