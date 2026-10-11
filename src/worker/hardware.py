from __future__ import annotations
import os
import platform
import shutil
import subprocess
from comum.protocolo import Hardware

def _ram_gb () -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.virtual_memory().total / 2**30, 1)

def _gpus_nvidia () -> list[str]:
    """
    Nome e memória das GPUs NVIDIA, se existirem.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        saida = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader"
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return [linha.strip() for linha in saida.splitlines() if linha.strip()]

def coletar_hardware () -> Hardware:
    return Hardware(
        sistema=platform.system(),
        nome_maquina=platform.node(),
        cpu=platform.processor() or platform.machine(),
        threads=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
        gpus=_gpus_nvidia(),
    )