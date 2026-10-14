from __future__ import annotations
import os
import platform
import shutil
import subprocess
import sys
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

def _nome_cpu () -> str:
    """O nome comercial do processador, como o Blender e o Open Data o mostram."""
    try:
        if sys.platform == "win32":
            import winreg
            chave = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, chave) as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=10, check=True).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as arquivo:
            for linha in arquivo:
                if linha.lower().startswith("model name"):
                    return linha.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()

def _nucleos_fisicos () -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.cpu_count(logical=False)

def _gpus_windows () -> list[str]:
    """Todas as placas de vídeo (NVIDIA, AMD, Intel), pelo WMI do Windows."""
    if sys.platform != "win32":
        return []
    try:
        saida = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return [linha.strip() for linha in saida.splitlines() if linha.strip()]

def coletar_hardware () -> Hardware:
    nvidia = _gpus_nvidia()
    outras = [g for g in _gpus_windows() if not any(g in n for n in nvidia)]
    return Hardware(
        sistema=platform.system(),
        nome_maquina=platform.node(),
        cpu=_nome_cpu(),
        threads=os.cpu_count() or 1,
        nucleos=_nucleos_fisicos(),
        ram_gb=_ram_gb(),
        gpus=nvidia + outras,
    )
