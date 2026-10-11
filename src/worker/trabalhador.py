"""
Sobe um no trabalhador.

python scr/worker/trabalhador.py --config config/trabalhador.yaml

Ciclo de vida:
1. coleta o hardware e se registra no mestre;
2. liga a thread de heartbeat;
3. reenvia os resultados que ficaram para trás (se caiu antes de enviar)
4. repete para sempre: pede job -> baixa a cena se precisar -> renderiza -> grava o relatório -> envia ao mestre -> apaga a pasta do job
"""

from __future__ import annotations
import argparse
import logging
import platform
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path[0] = str(Path(__file__).resolve().parents[1])

import httpx2
from comum.protocolo import Job, Progresso, Relatorio
from comum.util import agora_utc, escrever_json_atomico, sha256_arquivo, versao_tupla
from worker import blender
from worker.client import ClienteMestre, EntregaObsoleta, PrecisaRegistrar
from worker.config import ConfigTrabalhador, carregar_config
from worker.hardware import coletar_hardware
from worker.log_blender import AnaliseLog

log = logging.getLogger("renderopt.worker")

class Batimento (threading.Thread):
    """
    Thread que manda o heartbeat a cada certo intervalo de segundos.
    """
    def __init__(self, cfg: ConfigTrabalhador, intervalo: float):
        super().__init__(name="heartbeat", daemon=True)
        self.cliente = ClienteMestre(cfg.mestre, cfg.trabalhador_id, cfg.token)
        self.intervalo = intervalo
        self.job_atual = Optional[str] = None
        self.precisa_registrar = threading.Event()

    def run(self) -> None:
        while True:
            time.sleep(self.intervalo)
            try:
                self.cliente.heartbeat(self.job_atual)
            except PrecisaRegistrar:
                self.precisa_registrar.set()
            except Exception as erro:
                log.warning(f"Heartbeat falhou: {erro}")

def escolher_executavel (cfg: ConfigTrabalhador, versao_cena: str) -> Optional[list[str]]:
    """
    Escolhe o Blender de menor versão que ainda abre a cena.
    """
    compativeis = sorted(
        (v for v in cfg.blender if versao_tupla(v) >= versao_tupla(versao_cena)),
        key=versao_tupla
    )
    if not compativeis:
        return None
    executavel = cfg.blender[compativeis[0]]
    return [executavel] if isinstance(executavel, str) else list(executavel)

def garantir_cena (cliente: ClienteMestre, cfg: ConfigTrabalhador, job: Job) -> Path:
    """
    Cache local de cenas. Se a cena mudar no mestre, o hash dela muda e ela é
    baixada novamente, se não mudar o download acontece somente uma vez por
    máquina, não uma vez por frame.
    """
    caminho = cfg.pasta_cenas / f"{job.cena_sha256}.blend"
    if not caminho.is_file():
        log.info(f"Baixando a cena {job.cena_id}")
        cliente.baixar_cena(job, caminho)
    return caminho

def montar_relatorio (cfg: ConfigTrabalhador, job: Job, execucao: blender.Execucao, inicio_utc: str) -> Relatorio:
    analise, imagem = execucao.analise, execucao.imagem
    linhas = list(analise.ultimas_linhas)
    return Relatorio(
        job_id=job.job_id,
        token=job.token,
        trabalhador_id=cfg.trabalhador_id,
        status=execucao.status,
        motivo=execucao.motivo,
        cena_id=job.cena_id,
        frame=job.frame,
        dispositivo=cfg.dispositivo,
        sistema=platform.system(),
        versao_blender=analise.versao_blender,
        codigo_saida=execucao.codigo_saida,
        tempo_previsto_s=job.tempo_previsto_s,
        tempo_total_s=execucao.tempo_total_s,
        tempo_preparacao_s=analise.tempo_preparacao_s,
        tempo_amostragem_s=analise.tempo_amostragem_s,
        tempo_blender_s=analise.tempo_blender_s,
        tempo_gravacao_s=analise.tempo_gravacao_s,
        amostra_final=analise.amostra_atual,
        amostras_total=analise.amostras_total,
        instante_ultima_amostra_s=analise.instante_amostra_s,
        memoria_max_mb=analise.memoria_max_mb,
        imagem_nome=imagem.name if imagem else None,
        imagem_sha256=sha256_arquivo(imagem) if imagem else None,
        imagem_bytes=imagem.stat().st_size if imagem else None,
        inicio_utc=inicio_utc,
        fim_utc=agora_utc(),
        log_final=linhas[-10:] if execucao.status == "concluido" else linhas[-30:]
    )

def enviar_pendente (cliente: ClienteMestre, cfg: ConfigTrabalhador, pasta_job: Path) -> None:
    """
    Envia o relatório e a imagem de `pasta_job` e apaga a pasta.
    """
    relatorio = Relatorio.model_validate_json((pasta_job / "relatorio.json").read_text(encoding="utf-8"))
    imagem = pasta_job / "saida" / relatorio.imagem_nome if relatorio.imagem_nome else None
    try:
        cliente.enviar_resultado(relatorio, imagem)
        log.info(f"Job {relatorio.job_id}: {relatorio.status} em {relatorio.tempo_total_s:.1f} s (previsto {relatorio.tempo_previsto_s:.1f} s)")
    except EntregaObsoleta:
        log.info(f"Job {relatorio.job_id} já foi entregue a outra máquina, resultado descartado")
    except httpx2.HTTPStatusError as erro:
        log.error(f"Mestre recusou o resultado de {relatorio.job_id}: {erro.response.text}")
        destino = cfg.pasta_trabalho / "recusados" / pasta_job.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pasta_job), str(destino))
        return

    shutil.rmtree(pasta_job, ignore_errors=True)

def processar (cliente: ClienteMestre, cfg: ConfigTrabalhador, job: Job) -> None:
    pasta_job = cfg.pasta_jobs / f"{job.job_id}_{job.token[:8]}"
    shutil.rmtree(pasta_job, ignore_errors=True)
    inicio_utc = agora_utc()

    def falha (motivo: str) -> blender.Execucao:
        return blender.Execucao("falha", motivo, None, 0.0, AnaliseLog(), None)

    executavel = escolher_executavel(cfg, job.versao_cena)
    if executavel is None:
        execucao = falha(f"Nenhum Blender >= {job.versao_cena} configurado neste trabalhador.")
    else:
        try:
            cena = garantir_cena(cliente, cfg, job)
        except (ConnectionError, httpx2.HTTPError) as erro:
            execucao = falha(f"Não foi possível baixar a cena: {erro}")
        else:
            def consultar (decorrido: float, analise: AnaliseLog):
                progresso = Progresso(
                    amostra_atual=analise.amostra_atual,
                    amostras_total=analise.amostras_total,
                    tempo_preparacao_s=analise.tempo_preparacao_s,
                    instante_amostra_s=analise.instante_amostra_s,
                )
                return cliente.consultar_excesso(job, decorrido, progresso)

            comando = blender.montar_comando(executavel, cena, pasta_job / "saida", job.frame, cfg.dispositivo)
            execucao = blender.executar(
                comando,
                pasta_job / "saida",
                checkpoint_s=job.checkpoint_s,
                teto_s=job.teto_s,
                limite_sem_saida_s=job.limite_sem_saida_s,
                consultar=consultar,
            )

    relatorio = montar_relatorio(cfg, job, execucao, inicio_utc)
    escrever_json_atomico(pasta_job / "relatorio.json", relatorio.model_dump())
    enviar_pendente(cliente, cfg, pasta_job)

def main (argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="No trabalhador do render-opt.")
    parser.add_argument("--config", type=Path, required=True, help="Arquivo YAML de configuracao")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = carregar_config(args.config)
    cfg.pasta_jobs.mkdir(parents=True, exist_ok=True)

    cliente = ClienteMestre(cfg.mestre, cfg.trabalhador_id, cfg.token)
    hardware = coletar_hardware()
    hardware.fator_velocidade = cfg.fator_velocidade
    versoes = sorted(cfg.blender, key=versao_tupla)

    def registrar ():
        return cliente.registrar(hardware, versoes, cfg.dispositivo)

    parametros = registrar()
    log.info(f"Registrado em {cfg.mestre} como {cfg.trabalhador_id}")
    batimento = Batimento(cfg, parametros.intervalo_heartbeat_s)
    batimento.start()

    # Sobras de uma execução anterior, com relatório, reenvia, sem, o
    # job foi interrompido no meio e o mestre já o devolveu à fila
    for pasta in sorted(cfg.pasta_jobs.iterdir()):
        if (pasta / "relatorio.json").is_file():
            enviar_pendente(cliente, cfg, pasta)
        else:
            shutil.rmtree(pasta, ignore_errors=True)

    try:
        while True:
            if batimento.precisa_registrar.is_set():
                registrar()
                batimento.precisa_registrar.clear()

            espera = parametros.espera_longa_s
            try:
                # Com espera longa, esta chamada fica parada no mestre até surgir
                # trabalho (ou até `espera` segundos). O job chega na hora, sem
                # perguntar de novo a cada poucos segundos
                job = cliente.proximo_job(espera)
            except PrecisaRegistrar:
                batimento.precisa_registrar.set()
                continue

            if job is None:
                if espera <= 0: # Mestre sem espera longa
                    time.sleep(parametros.intervalo_ocioso_s)
                continue

            log.info(f"Job {job.job_id} (frame {job.frame}): previsto {job.tempo_previsto_s} s, checkpoint {f'{job.checkpoint_s:.0f}' if job.checkpoint_s else 'nenhum'}")
            batimento.job_atual = job.job_id
            try:
                processar(cliente, cfg, job)
            finally:
                batimento.job_atual = None

    except KeyboardInterrupt:
        # Aqui, o Blender já foi finalizado pelo finally de blender.executar()
        log.info("Encerrado pelo operador")

    return 0

if __name__ == "__main__":
    sys.exit(main())