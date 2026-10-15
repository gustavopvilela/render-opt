from __future__ import annotations
from typing import Annotated, Literal, Optional
from pydantic import AfterValidator, BaseModel, Field
from comum.enums import StatusRelatorio
from comum.util import validar_id

Id = Annotated[str, AfterValidator(validar_id)]
Versao = Annotated[str, Field(pattern=r"^\d+\.\d+$")]

class Hardware (BaseModel):
    sistema: str
    nome_maquina: str
    cpu: str
    threads: int
    nucleos: Optional[int] = None # Núcleos físicos
    ram_gb: Optional[float] = None
    gpus: list[str] = []
    fator_velocidade: Optional[float] = None

class Registro (BaseModel):
    trabalhador_id: Id
    hardware: Hardware
    versoes_blender: list[Versao]
    dispositivo: str

class RespostaRegistro (BaseModel):
    intervalo_heartbeat_s: float
    intervalo_ocioso_s: float
    espera_longa_s: float = 0

class Heartbeat (BaseModel):
    job_atual: Optional[str] = None

class Job (BaseModel):
    job_id: Id
    token: str
    lote_id: Id
    cena_id: Id
    cena_sha256: str
    versao_cena: Versao
    frame: int
    tempo_previsto_s: float
    checkpoint_s: Optional[float] # Quando perguntar ao mestre se continua; None = nunca
    teto_s: Optional[float] # Tempo máximo de processamento, após isso o processo é encerrado e declarado como falha
    limite_sem_saida_s: float # Blender não responde por mais que isso, é considerado como travado

class Progresso (BaseModel):
    amostra_atual: Optional[int] = None
    amostras_total: Optional[int] = None
    tempo_preparacao_s: Optional[float] = None
    instante_amostra_s: Optional[float] = None

class ConsultaExcesso (BaseModel):
    token: str
    tempo_decorrido_s: float
    progresso: Progresso

class DecisaoExcesso (BaseModel):
    acao: Literal["continuar", "abortar"]
    motivo: str
    proximo_checkpoint_s: Optional[float] = None
    novo_teto_s: Optional[float] = None # Se continuar, o teto é aumentado

class Relatorio (BaseModel):
    """
    JSON que o trabalhador envia ao terminar um job, com ou sem sucesso
    """
    job_id: Id
    token: str
    trabalhador_id: Id
    status: StatusRelatorio
    motivo: Optional[str] = None

    cena_id: Id
    frame: int
    dispositivo: str
    sistema: str
    versao_blender: Optional[str] = None
    codigo_saida: Optional[int] = None

    tempo_previsto_s: float
    tempo_total_s: float # Do lançamento do Blender até terminar
    tempo_preparacao_s: Optional[float] = None # Até aparecer a linha Sample 0/N
    tempo_amostragem_s: Optional[float] = None # De Sample 0/N até Saved
    tempo_blender_s: Optional[float] = None # Tempo que o próprio Blender calcula
    tempo_gravacao_s: Optional[float] = None
    amostra_final: Optional[int] = None
    amostras_total: Optional[int] = None
    instante_ultima_amostra_s: Optional[float] = None
    memoria_max_mb: Optional[float] = None

    imagem_nome: Optional[str] = None
    imagem_sha256: Optional[str] = None
    imagem_bytes: Optional[int] = None

    inicio_utc: str
    fim_utc: str
    log_final: list[str] = [] # Ultimas linhas do Blender, para diagnóstico

class CenaDoLote (BaseModel):
    cena_id: Id
    cena_sha256: str
    versao_cena: Versao
    frames: list[int] = Field(min_length=1)

class NovoLote (BaseModel):
    lote_id: Id
    cenas: list[CenaDoLote] = Field(min_length=1)
    distribuicao: Literal["sa", "flamenco"] = "sa"