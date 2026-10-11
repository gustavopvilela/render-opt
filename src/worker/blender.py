"""
Executa o Blender para um frame, vigiando tempo e progresso.
"""
from __future__ import annotations
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from comum.protocolo import DecisaoExcesso
from worker.log_blender import AnaliseLog

_FIM = object()     # marcador: o Blender fechou a saída

@dataclass
class Execucao:
    status: str     # "concluido" | "abortado" | "falha"
    motivo: Optional[str]
    codigo_saida: Optional[int]
    tempo_total_s: float
    analise: AnaliseLog
    imagem: Optional[Path]

# Recebe (segundos decorridos, analise) e devolve a decisão ao mestre,
# ou None se o mestre não respondeu
Consulta = Callable[[float, AnaliseLog], Optional[DecisaoExcesso]]

def montar_comando (executavel: list[str], cena: Path, pasta_saida: Path, frame: int, dispositivo: str) -> list[str]:
    return [
        *executavel,
        "--factory-startup", "-noaudio",
        "-b", str(cena),
        "-F", "PNG", "-x", "1",
        "-o", str(pasta_saida / "frame_######"),
        "-f", str(frame),
        "--", "--cycles-device", dispositivo,
    ]

def _iniciar (comando: list[str]) -> subprocess.Popen:
    extras: dict = {}
    if os.name == "nt":
        extras["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        extras["start_new_session"] = True
    return subprocess.Popen(
        comando,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **extras
    )

def _matar (processo: subprocess.Popen) -> None:
    """
    Encerra o Blender e seus filhos.
    """
    if processo.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            [
                "taskkill",
                "/PID",
                str(processo.pid),
                "/T",
                "/F"
            ],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(processo.pid, signal.SIGTERM)
            processo.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(processo.pid, signal.SIGKILL)
    processo.wait()

def _ler (processo: subprocess.Popen, fila: queue.Queue, inicio: float) -> None:
    """
    Thread leitora, cada linha vai para a fila junto do instante em que chegou.
    """
    for linha in processo.stdout:
        fila.put((time.perf_counter() - inicio, linha.rstrip("\r\n")))
    fila.put((time.perf_counter() - inicio, _FIM))

def executar (
    comando: list[str],
    pasta_saida: Path,
    *,
    checkpoint_s: Optional[float],
    teto_s: Optional[float],
    limite_sem_saida_s: float,
    consultar: Consulta,
) -> Execucao:
    """
    Roda o Blender e devolve o resultado. Há 3 marcos:

    - checkpoint_s: ao passar dele, pergunta ao mestre se continua
    - teto_s: limite rígido, se passo, mata sem perguntar
    - limite_sem_saida_s: Blender sem escrever por esse tempo é considerado travado
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)
    analise = AnaliseLog()
    linhas: queue.Queue = queue.Queue()
    inicio = time.perf_counter()
    processo = _iniciar(comando)
    threading.Thread(
        target=_ler,
        args=(processo, linhas, inicio),
        daemon=True,
    ).start()

    ultimo_sinal = 0.0  # Instante da última linha recebida
    proximo_checkpoint = checkpoint_s
    status: Optional[str] = None
    motivo: Optional[str] = None

    try:
        while True:
            try:
                instante, linha = linhas.get(timeout=0.5)
            except queue.Empty:
                instante, linha = None, None

            if linha is _FIM:
                break
            if linha is not None:
                ultimo_sinal = instante
                analise.alimentar(linha, instante)

            decorrido = time.perf_counter() - inicio
            if teto_s is not None and decorrido > teto_s:
                status, motivo = "falha", f"Passou do teto de {teto_s:.0f} s"
            elif decorrido - ultimo_sinal > limite_sem_saida_s:
                status, motivo = "falha", f"Blender sem escrever nada há {limite_sem_saida_s:.0f} s"
            elif proximo_checkpoint is not None and decorrido > proximo_checkpoint:
                decisao = consultar(decorrido, analise)
                if decisao is None:
                    # Mestre fora do alcance, tenta de novo em 1 minuto
                    proximo_checkpoint = decorrido + 60
                elif decisao.acao == "abortar":
                    status, motivo = "abortado", decisao.motivo
                else:
                    proximo_checkpoint = decisao.proximo_checkpoint_s
                    if teto_s is not None and decisao.novo_teto_s is not None:
                        teto_s = max(teto_s, decisao.novo_teto_s)

            if status is not None:
                _matar(processo)
                break
    finally:
        # Saída por excessão, o Blender não pode ficar órfão
        _matar(processo)

    tempo_total = time.perf_counter() - inicio
    codigo = processo.returncode
    imagens = sorted(pasta_saida.glob("frame_*.png"))

    if status is None:
        if codigo == 0 and len(imagens) == 1:
            status = "concluido"
        else:
            status = "falha"
            motivo = f"Código de saída {codigo}, {len(imagens)} imagem(ns) gerada(s)"

    return Execucao(
        status=status,
        motivo=motivo,
        codigo_saida=codigo,
        tempo_total_s=tempo_total,
        analise=analise,
        imagem=imagens[0] if status == "concluido" else None,
    )