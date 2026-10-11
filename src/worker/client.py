"""
Conversa com o mestre por HTTP.

Toda chamada tem timeout. Falha de rede ou erro 5xx do mestre é
tratado como passageiro: tenta de novo com espera exponencial
e um pouco de aleatoriedade ("jitter"), para que vários trabalhadores
não sobrecarreguem o mestre no mesmo instante quando ele volta.
"""
from __future__ import annotations
import hashlib
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional
import httpx
from comum.protocolo import (
    ConsultaExcesso, DecisaoExcesso, Hardware, Heartbeat, Job, Progresso,
    Registro, Relatorio, RespostaRegistro
)

log = logging.getLogger("renderopt.worker")


class PrecisaRegistrar(Exception):
    """O mestre não reconhece esse trabalhador."""


class EntregaObsoleta(Exception):
    """O mestre já deu este job a outra máquina, o resultado deve ser descartado."""


class ClienteMestre:
    def __init__(self, url_mestre: str, trabalhador_id: str, token: Optional[str]):
        self.trabalhador_id = trabalhador_id
        self.http = httpx.Client(
            base_url=url_mestre,
            headers={"X-Token": token} if token else {},
            timeout=httpx.Timeout(10.0, read=60.0),
        )

    def _pedir(self, fazer: Callable[[], httpx.Response], tentativas: Optional[int] = None) -> httpx.Response:
        """
        Executa `fazer()` repetindo em falha de rede ou 5xx.
        :param tentativas: se igual a None, insiste para sempre.
        """
        espera, n = 1.0, 0
        while True:
            try:
                resposta = fazer()
                if resposta.status_code < 500:
                    return resposta
                problema = f"Mestre respondeu {resposta.status_code}"
            except httpx.TransportError as erro:
                problema = f"{type(erro).__name__}: {erro}"

            n += 1
            if tentativas is not None and n >= tentativas:
                raise ConnectionError(problema)

            pausa = random.uniform(0, espera)
            log.warning(f"{problema}; nova tentativa em {pausa:.1f}")
            time.sleep(pausa)
            espera = min(espera * 2, 60.0)

    @staticmethod
    def _checar(resposta: httpx.Response) -> httpx.Response:
        if resposta.status_code == 404 and "registre-se" in resposta.text:
            raise PrecisaRegistrar()
        if resposta.status_code == 409:
            raise EntregaObsoleta(resposta.text)
        resposta.raise_for_status()
        return resposta

    def registrar(self, hardware: Hardware, versoes: list[str], dispositivo: str) -> RespostaRegistro:
        corpo = Registro(trabalhador_id=self.trabalhador_id, hardware=hardware, versoes_blender=versoes,
                         dispositivo=dispositivo)
        resposta = self._pedir(lambda: self.http.post("/trabalhadores/registrar", json=corpo.model_dump(mode="json")))
        return RespostaRegistro.model_validate_json(self._checar(resposta).content)

    def heartbeat(self, job_atual: Optional[str]) -> None:
        corpo = Heartbeat(job_atual=job_atual).model_dump(mode="json")
        resposta = self._pedir(lambda: self.http.post(f"/trabalhadores/{self.trabalhador_id}/heartbeat", json=corpo),
                               tentativas=1)
        self._checar(resposta)

    def proximo_job(self, espera_s: float = 0) -> Optional[Job]:
        """
        :param espera_s: Se maior que 0, o mestre segura a resposta até surgir um job.
        """
        resposta = self._checar(
            self._pedir(
                lambda: self.http.post(
                    f"/trabalhadores/{self.trabalhador_id}/proximo-job",
                    params={"espera_s": espera_s},
                    timeout=httpx.Timeout(10.0, read=espera_s + 30.0)
                )
            )
        )
        if resposta.status_code == 204:
            return None
        return Job.model_validate_json(resposta.content)

    def consultar_excesso(self, job: Job, decorrido: float, progresso: Progresso) -> Optional[DecisaoExcesso]:
        """
        :return: None se o mestre não responder, quem chamou decide o que fazer.
        """
        corpo = ConsultaExcesso(
            token=job.token,
            tempo_decorrido_s=decorrido,
            progresso=progresso,
        )
        try:
            resposta = self._checar(
                self._pedir(
                    lambda: self.http.post(
                        f"/jobs/{job.job_id}/excesso",
                        json=corpo.model_dump(mode="json"),
                    ),
                    tentativas=3,
                )
            )
        except EntregaObsoleta:
            return DecisaoExcesso(
                acao="abortar",
                motivo="O mestre já reatribuiu este job"
            )
        except (ConnectionError, httpx.HTTPStatusError):
            return None
        return DecisaoExcesso.model_validate_json(resposta.content)

    def enviar_resultado(self, relatorio: Relatorio, imagem: Optional[Path]) -> None:
        """
        Envia o JSON final e, se houver, a imagem. Insiste até o mestre responder.
        """

        def fazer() -> httpx.Response:
            dados = {"relatorio": relatorio.model_dump_json()}
            if imagem is None:
                return self.http.post(
                    f"/jobs/{relatorio.job_id}/resultado",
                    data=dados
                )
            with imagem.open("rb") as arquivo:
                return self.http.post(
                    f"/jobs/{relatorio.job_id}/resultado",
                    data=dados,
                    files={"imagem": (imagem.name, arquivo, "image/png")},
                    timeout=httpx.Timeout(10.0, write=300.0, read=60.0)
                )

        self._checar(self._pedir(fazer))

    def baixar_cena(self, job: Job, destino: Path) -> None:
        """
        Baixa o .blend em blocos, conferindo o SHA-256. Aparece na pasta destino
        se chegar inteiro e correto (escrita em .parcial + os.replace)
        """
        destino.parent.mkdir(parents=True, exist_ok=True)
        parcial = destino.with_name(destino.name + ".parcial")

        def fazer() -> httpx.Response:
            h = hashlib.sha256()
            with self.http.stream(
                    "GET",
                    f"/cenas/{job.cena_id}",
                    timeout=httpx.Timeout(10.0, read=300.0)
            ) as resposta:
                if resposta.status_code != 200:
                    resposta.read()
                    return resposta
                with parcial.open("wb") as arquivo:
                    for bloco in resposta.iter_bytes(1 << 20):
                        arquivo.write(bloco)
                        h.update(bloco)

            if h.hexdigest() != job.cena_sha256:
                parcial.unlink(missing_ok=True)
                raise httpx.ReadError("SHA-256 de cena não confere (download corrompido)")
            return resposta

        self._checar(self._pedir(fazer, tentativas=5))
        os.replace(parcial, destino)