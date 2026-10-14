"""
Atributos de hardware a partir do NOME do dispositivo.

Numa validação agrupada por máquina, o nome da máquina de teste nunca aparece
no treino, então o nome cru não serve como atributo. O que generaliza é o que
máquinas diferentes compartilham: fabricante, família, ano da arquitetura,
faixa dentro da geração e se é de notebook.

Regra de ouro: na dúvida, devolver None. Um None vira NaN, que o LightGBM
trata sozinho. Um número plausível e errado engana o modelo em silêncio.
"""

from __future__ import annotations
import re
from typing import Optional

# Ano aproximado de cada geração
ANO_GEFORCE = {6: 2012, 7: 2013, 8: 2014, 9: 2014, 10: 2016, 16: 2019, 20: 2018, 30: 2020, 40: 2022, 50: 2025}
ANO_GEFORCE_MX = {1: 2017, 2: 2019, 3: 2020, 4: 2020, 5: 2021}
ANO_NVIDIA_LETRA = {"K": 2012, "M": 2014, "P": 2016, "V": 2017, "T": 2018, "A": 2020, "L": 2022, "H": 2022, "B": 2025}
ANO_TITAN = {"TITAN RTX": 2018, "TITAN V": 2017, "TITAN XP": 2017, "TITAN X (PASCAL)": 2016,
             "TITAN X": 2015, "TITAN BLACK": 2014, "TITAN": 2013}
ANO_RADEON_RX_3 = {4: 2016, 5: 2017}                       # RX 480, RX 580
ANO_RADEON_RX_4 = {5: 2019, 6: 2020, 7: 2022, 9: 2025}     # RX 5700, 6800, 7900, 9070
ANO_INTEL_CORE = {1: 2010, 2: 2011, 3: 2012, 4: 2013, 5: 2015, 6: 2015, 7: 2017, 8: 2017, 9: 2018,
                  10: 2020, 11: 2021, 12: 2021, 13: 2022, 14: 2023}
ANO_RYZEN = {1: 2017, 2: 2018, 3: 2019, 4: 2020, 5: 2020, 6: 2022, 7: 2022, 8: 2024, 9: 2024}
ANO_APPLE = {1: 2020, 2: 2022, 3: 2023, 4: 2024, 5: 2025}
NIVEL_APPLE = {None: 1, "PRO": 2, "MAX": 3, "ULTRA": 4}
NIVEL_XEON = {"BRONZE": 1, "SILVER": 2, "GOLD": 3, "PLATINUM": 4}

_LIXO = re.compile(r"\((?:R|TM)\)|®|™|\bCPU\b|\bPROCESSOR\b|\[ZLUDA\]|�|@.*$")


def normalizar (nome: Optional[str]) -> str:
    """Maiúsculas, sem marcas registradas, sem o clock ("@ 2.60GHz") e sem espaços repetidos."""
    if not isinstance(nome, str):
        return ""
    return " ".join(_LIXO.sub(" ", nome.upper()).split())


def _vazio () -> dict:
    return {
        "hw_fabricante": None,
        "hw_familia": "OUTRO",
        "hw_ano": None,
        "hw_classe": None,
        "hw_notebook": None,
        "hw_nucleos_gpu": None,
        "hw_vram_gb": None,
    }


def _fabricante (n: str) -> Optional[str]:
    if re.search(r"NVIDIA|GEFORCE|QUADRO|TESLA|TITAN|\bRTX\b|\bGTX\b|\bGRID\b", n):
        return "NVIDIA"
    if re.search(r"\bAMD\b|RADEON|RYZEN|EPYC|THREADRIPPER|\bFX\b|PHENOM|ATHLON|NAVI", n):
        return "AMD"
    if re.search(r"INTEL|XEON|\bCORE\b|CELERON|PENTIUM|\bARC\b|IRIS", n):
        return "INTEL"
    if re.search(r"\bAPPLE\b", n):
        return "APPLE"
    return None


# NVIDIA

def _nvidia (n: str, f: dict) -> None:
    f["hw_notebook"] = bool(re.search(r"LAPTOP|MAX-Q|MOBILE|\d{3,4}MX?\b", n))

    for chave, ano in ANO_TITAN.items():
        if chave in n:
            f["hw_familia"], f["hw_ano"], f["hw_classe"] = "TITAN", ano, 99
            return
    # Profissionais (antes da GeForce)
    if m := re.search(r"\bRTX\s+(?:PRO\s+)?(\d{3,4})\b.*\b(ADA|BLACKWELL)\b", n):
        f["hw_familia"] = "NVIDIA_PRO"
        f["hw_ano"] = 2022 if m[2] == "ADA" else 2025
        f["hw_classe"] = int(m[1]) // 1000
        return
    if m := re.search(r"QUADRO\s+RTX\s+(\d)\d{3}", n):
        f["hw_familia"], f["hw_ano"], f["hw_classe"] = "NVIDIA_PRO", 2018, int(m[1])
        return
    if m := re.search(r"(?:QUADRO|TESLA|GRID|NVIDIA|RTX)\s+([KMPVTALHB])(\d{1,4})(?!\d)", n):
        f["hw_familia"] = "NVIDIA_PRO"
        f["hw_ano"] = ANO_NVIDIA_LETRA[m[1]]
        f["hw_classe"] = int(m[2][0])
        return
    # GeForce
    if m := re.search(r"\bMX\s?(\d)(\d{2})\b", n):
        f["hw_familia"], f["hw_ano"], f["hw_classe"] = "GEFORCE_MX", ANO_GEFORCE_MX.get(int(m[1])), int(m[2])
        f["hw_notebook"] = True
        return
    if m := re.search(r"\b(RTX|GTX|GT)\s?(\d{3,4})(?:M|TI|S)?\b", n):
        num = int(m[2])
        f["hw_familia"] = "GEFORCE_RTX" if m[1] == "RTX" else "GEFORCE_GTX"
        f["hw_ano"] = ANO_GEFORCE.get(num // 100)
        f["hw_classe"] = num % 100
        return
    if m := re.search(r"GEFORCE\s+(\d)(\d{2})MX?\b", n):   # GeForce 940MX, 840M
        f["hw_familia"] = "GEFORCE_GTX"
        f["hw_ano"] = ANO_GEFORCE.get(int(m[1]))
        f["hw_classe"] = int(m[2])
        f["hw_notebook"] = True


# AMD

def _amd_gpu (n: str, f: dict) -> None:
    f["hw_notebook"] = bool(re.search(r"\b\d{3,4}[MS]\b", n))
    if m := re.search(r"\bRX\s?(\d{3,4})", n):
        num = int(m[1])
        f["hw_familia"] = "RADEON_RX"
        if num >= 1000:
            f["hw_ano"] = ANO_RADEON_RX_4.get(num // 1000)
            f["hw_classe"] = (num // 100) % 10
        else:
            f["hw_ano"] = ANO_RADEON_RX_3.get(num // 100)
            f["hw_classe"] = (num // 10) % 10
        return
    if m := re.search(r"\bAI\s+PRO\s+R(\d)\d{3}", n):  # Radeon AI PRO R9700
        f["hw_familia"], f["hw_ano"], f["hw_classe"] = "RADEON_PRO", 2025, int(m[1])
        return
    if m := re.search(r"\bPRO\s+W(\d)(\d)\d{2}", n):   # Pro W6800, W7900
        f["hw_familia"] = "RADEON_PRO"
        f["hw_ano"] = {5: 2019, 6: 2020, 7: 2023, 9: 2025}.get(int(m[1]))
        f["hw_classe"] = int(m[2])
        return
    if m := re.search(r"\bPRO\s+(\d)(\d)\d(\d)?(?!\d)", n):  # Mac: Pro 5500M (2019), Pro 560X (2017)
        f["hw_familia"] = "RADEON_PRO"
        f["hw_ano"] = 2019 if m[3] else 2017
        f["hw_classe"] = int(m[2])
        return
    if m := re.search(r"\b\d0(\d)0S\b", n):            # 8060S (Strix Halo)
        f["hw_familia"], f["hw_ano"], f["hw_classe"] = "RADEON_IGPU", 2025, int(m[1])
        f["hw_notebook"] = True
        return
    if m := re.search(r"\b([678])(\d)0M\b", n):        # 680M, 780M, 880M
        f["hw_familia"] = "RADEON_IGPU"
        f["hw_ano"] = {6: 2022, 7: 2023, 8: 2024}[int(m[1])]
        f["hw_classe"] = int(m[2])
        f["hw_notebook"] = True
        return
    if "VEGA" in n or "RADEON VII" in n:
        f["hw_familia"] = "RADEON_VEGA"
        f["hw_ano"] = 2019 if "VII" in n else 2017


def _amd_cpu (n: str, f: dict) -> None:
    if m := re.search(r"THREADRIPPER\s+(?:PRO\s+)?(\d)\d{3}", n):
        f["hw_familia"], f["hw_classe"] = "THREADRIPPER", 9
        f["hw_ano"] = {1: 2017, 2: 2018, 3: 2019, 5: 2022, 7: 2023, 9: 2025}.get(int(m[1]))
        return
    if m := re.search(r"EPYC\s+\d[\dA-Z]{2}(\d)", n):  # último dígito = geração (7302, 7R13)
        f["hw_familia"], f["hw_classe"] = "EPYC", 9
        f["hw_ano"] = {1: 2017, 2: 2019, 3: 2021, 4: 2022, 5: 2024}.get(int(m[1]))
        return
    if m := re.search(r"RYZEN\s+(?:PRO\s+)?(\d)\s+(?:PRO\s+)?(\d)\d{3}([A-Z0-9]*)", n):
        f["hw_familia"] = "RYZEN"
        f["hw_ano"] = ANO_RYZEN.get(int(m[2]))
        f["hw_classe"] = int(m[1])
        f["hw_notebook"] = m[3][:1] in ("H", "U", "S")
        return
    if "RYZEN AI" in n or re.search(r"RYZEN\s+\d\s+\d{3}\b", n):   # Ryzen AI 9 HX 370, Ryzen 9 270
        f["hw_familia"], f["hw_ano"] = "RYZEN", 2024
        return
    if re.search(r"\bFX\b|PHENOM|ATHLON|\bA\d{1,2}-", n):
        f["hw_familia"], f["hw_ano"] = "AMD_ANTIGA", 2011


# Intel

def _intel_cpu (n: str, f: dict) -> None:
    if m := re.search(r"CORE\s+ULTRA\s+(\d)\s+(\d)\d{2}([A-Z]*)", n):
        f["hw_familia"] = "CORE_ULTRA"
        f["hw_ano"] = 2023 if m[2] == "1" else 2024
        f["hw_classe"] = int(m[1])
        f["hw_notebook"] = m[3][:1] in ("H", "U", "V")
        return
    if m := re.search(r"\bI([3579])-(\d{4,5})([A-Z]*)", n):
        digitos, sufixo = m[2], m[3]
        if len(digitos) == 5 or (digitos[0] == "1" and sufixo[:1] in ("G", "N")):
            geracao = int(digitos[:2])               # i5-12490F, i7-1065G7, i5-1135G7
        else:
            geracao = int(digitos[0])                # i7-3770K
        if g := re.search(r"(\d{1,2})TH GEN", n):    # "12th Gen Intel Core..." manda
            geracao = int(g[1])
        f["hw_familia"] = "CORE_I"
        f["hw_ano"] = ANO_INTEL_CORE.get(geracao)
        f["hw_classe"] = int(m[1])
        f["hw_notebook"] = sufixo[:1] in ("H", "U", "M", "Y", "P", "G", "N")
        return
    if m := re.search(r"XEON\s+(GOLD|PLATINUM|SILVER|BRONZE)\s+\d(\d)\d{2}", n):
        f["hw_familia"] = "XEON_SCALABLE"
        f["hw_ano"] = {1: 2017, 2: 2019, 3: 2021, 4: 2023, 5: 2023}.get(int(m[2]))
        f["hw_classe"] = NIVEL_XEON[m[1]]
        return
    if m := re.search(r"XEON\s+W([3579])-\d{4}", n):    # w5-3435X (2023)
        f["hw_familia"], f["hw_ano"], f["hw_classe"] = "XEON_W", 2023, int(m[1])
        return
    if m := re.search(r"XEON\s+W-(\d)(\d)\d{2,3}", n):  # W-2150B, W-3275M, W-10885M
        f["hw_familia"] = "XEON_W"
        f["hw_ano"] = 2020 if m[1] == "1" else {1: 2017, 2: 2019, 3: 2021}.get(int(m[2]))
        return
    if m := re.search(r"XEON\s+E([357])-\s?\d{4}[A-Z]?\s*(?:V(\d))?", n):
        f["hw_familia"] = "XEON_E"
        f["hw_ano"] = {1: 2012, 2: 2013, 3: 2014, 4: 2016, 5: 2015, 6: 2017}.get(int(m[2]) if m[2] else 1)
        f["hw_classe"] = int(m[1])
        return
    if re.search(r"XEON\s+[XLEW]\d{4}\b|CORE2|CORE\s+I7\s+\d{3}\b", n):
        f["hw_familia"], f["hw_ano"] = "INTEL_ANTIGA", 2009
        return
    if "CELERON" in n or "PENTIUM" in n:
        f["hw_familia"], f["hw_classe"] = "INTEL_ENTRADA", 1


def _intel_gpu (n: str, f: dict) -> None:
    if m := re.search(r"\bARC\s+(?:PRO\s+)?([AB])(\d)\d{1,2}(M?)\b", n):
        f["hw_familia"] = "ARC"
        f["hw_ano"] = 2022 if m[1] == "A" else 2024
        f["hw_classe"] = int(m[2])
        f["hw_notebook"] = m[3] == "M"
        return
    if m := re.search(r"\bARC\s+(?:PRO\s+)?1([34])0([VT])\b", n):   # gráficos integrados
        f["hw_familia"] = "ARC_IGPU"
        f["hw_ano"] = 2024 if m[2] == "V" else 2025
        f["hw_classe"] = int(m[1])
        f["hw_notebook"] = True
        return
    if "IRIS" in n:
        f["hw_familia"] = "INTEL_IGPU"
        f["hw_ano"] = 2020 if "XE" in n else 2015
        f["hw_notebook"] = True


# Apple

def _apple (n: str, f: dict) -> None:
    if m := re.search(r"\bM(\d)\s*(PRO|MAX|ULTRA)?\b", n):
        f["hw_familia"] = "APPLE_M"
        f["hw_ano"] = ANO_APPLE.get(int(m[1]))
        f["hw_classe"] = NIVEL_APPLE[m[2]]
    if m := re.search(r"(\d+)\s+CORES\)", n):
        f["hw_nucleos_gpu"] = int(m[1])


def features_hardware (nome: Optional[str], tipo: Optional[str]) -> dict:
    """
    Atributos estruturados de um dispositivo.
    :param nome: nome como o Blender reporta ("NVIDIA GeForce RTX 3060 Laptop GPU")
    :param tipo: CPU, CUDA, OPTIX, HIP, METAL ou ONEAPI
    """
    f = _vazio()
    n = normalizar(nome)
    if not n:
        return f
    f["hw_fabricante"] = _fabricante(n)
    if m := re.search(r"(\d+)\s*GB\b", n):
        f["hw_vram_gb"] = float(m[1])

    eh_cpu = str(tipo).upper() == "CPU"
    analisar = {
        ("APPLE", True): _apple, ("APPLE", False): _apple,
        ("INTEL", True): _intel_cpu, ("INTEL", False): _intel_gpu,
        ("AMD", True): _amd_cpu, ("AMD", False): _amd_gpu,
        ("NVIDIA", False): _nvidia,
    }.get((f["hw_fabricante"], eh_cpu))
    if analisar is not None:
        analisar(n, f)
    return f

# Máquina da produção

def versao_major_minor (versao: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """'5.1' -> (5, 1); qualquer coisa ilegível -> (None, None)."""
    m = re.match(r"\s*(\d+)\.(\d+)", str(versao or ""))
    return (int(m[1]), int(m[2])) if m else (None, None)


FABRICANTE_DO_BACKEND = {"CUDA": "NVIDIA", "OPTIX": "NVIDIA", "HIP": "AMD", "ONEAPI": "INTEL", "METAL": "APPLE"}


def nome_dispositivo (dispositivo: str, hardware: dict) -> Optional[str]:
    """
    O nome que o Blender reportaria para o dispositivo de render.
    CPU e METAL: o nome do processador (num Mac, a GPU é parte do chip: "Apple M2 Max").
    Outras GPUs: a primeira placa da ficha do fabricante do backend (num notebook com
    Intel UHD e uma GeForce, o OPTIX é a GeForce), sem o ', 4096 MiB' do nvidia-smi.
    """
    tipo = str(dispositivo).upper()
    if tipo in ("CPU", "METAL"):
        return hardware.get("cpu") or None
    gpus = [g.split(",")[0].strip() for g in hardware.get("gpus") or [] if g and g.strip()]
    do_fabricante = [g for g in gpus if _fabricante(normalizar(g)) == FABRICANTE_DO_BACKEND.get(tipo)]
    escolhidas = do_fabricante or gpus
    return escolhidas[0] if escolhidas else None


def linha_maquina (dispositivo: str, hardware: dict, versao_blender: Optional[str],
                   trabalhador_id: Optional[str] = None) -> dict:
    """
    As colunas de máquina que os modelos usam, a partir da ficha do trabalhador
    (comum.protocolo.Hardware como dict). Mesmas colunas e mesmas regras do
    opendata.acrescentar_hardware, para o treino e a produção descreverem a
    máquina do mesmo jeito.

    :param versao_blender: a versão que renderiza (ou vai renderizar) o frame, "5.1"
    """
    tipo = str(dispositivo).upper()
    nome = nome_dispositivo(tipo, hardware)
    host = features_hardware(hardware.get("cpu"), "CPU")
    major, minor = versao_major_minor(versao_blender)
    maquina = f"{normalizar(nome)}|{tipo}" if nome else f"TRABALHADOR {trabalhador_id}|{tipo}"
    return {
        "maquina": maquina,
        "dispositivo_nome": nome,
        "dispositivo_tipo": tipo,
        "cpu_nome": hardware.get("cpu"),
        "num_cpu_cores": hardware.get("nucleos"),
        "num_cpu_threads": hardware.get("threads"),
        "num_cpu_sockets": hardware.get("soquetes") or 1,
        "sistema": hardware.get("sistema"),
        "versao_blender_major": major,
        "versao_blender_minor": minor,
        **features_hardware(nome, tipo),
        "host_familia": host["hw_familia"], "host_ano": host["hw_ano"], "host_classe": host["hw_classe"],
    }
