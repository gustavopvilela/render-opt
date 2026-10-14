"""
Atributos dos frames de uma cena, extraídos no MESTRE na hora de planejar.

Roda src/extract/extract_scene.py dentro do Blender e guarda o resultado num
cache com o nome igual ao SHA-256 da cena: a mesma cena nunca é extraída duas
vezes, nem depois de reiniciar o mestre.
"""

from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence
import pandas as pd
from comum.util import sha256_arquivo, versao_tupla

EXTRATOR = Path(__file__).resolve().parents[1] / "extract" / "extract_scene.py"


def intervalos (frames: Sequence[int]) -> str:
    """[1, 2, 3, 7, 9, 10] -> "1-3,7,9-10". Mantém a linha de comando curta com milhares de frames."""
    partes, ordenados = [], sorted(set(frames))
    inicio = anterior = ordenados[0]
    for f in ordenados[1:] + [None]:
        if f is not None and f == anterior + 1:
            anterior = f
            continue
        partes.append(str(inicio) if inicio == anterior else f"{inicio}-{anterior}")
        if f is not None:
            inicio = anterior = f
    return ",".join(partes)


def escolher_blender (blenders: dict[str, str], versao: str) -> str:
    """O Blender de menor versão que abre a cena (mesma regra do submit.py)."""
    compativeis = sorted((v for v in blenders if versao_tupla(v) >= versao_tupla(versao)), key=versao_tupla)
    if not compativeis:
        raise RuntimeError(f"Nenhum Blender do mestre abre uma cena {versao}")
    return blenders[compativeis[0]]


def extrair (blender: str, blend: Path, frames: Sequence[int], pasta_cache: Path,
             timeout_s: float = 1800) -> pd.DataFrame:
    """Uma linha por frame pedido, na ordem pedida."""
    pasta_cache.mkdir(parents=True, exist_ok=True)
    cache = pasta_cache / f"{sha256_arquivo(blend)}.jsonl"
    linhas: dict[int, dict] = {}
    if cache.exists():
        for texto in cache.read_text(encoding="utf-8").splitlines():
            if texto.strip():
                r = json.loads(texto)
                linhas[int(r["frame"])] = r

    if faltam := sorted(set(frames) - set(linhas)):
        with tempfile.TemporaryDirectory() as tmp:
            saida = Path(tmp) / "features.jsonl"
            subprocess.run(
                [blender, "-b", str(blend), "--factory-startup", "-noaudio",
                 "-P", str(EXTRATOR), "--", "--saida", str(saida), "--frames", intervalos(faltam)],
                check=True, capture_output=True, timeout=timeout_s,
            )
            novas = [json.loads(t) for t in saida.read_text(encoding="utf-8").splitlines() if t.strip()]
        with cache.open("a", encoding="utf-8") as f:
            for r in novas:
                linhas[int(r["frame"])] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if faltam := [f for f in frames if f not in linhas]:
        raise RuntimeError(f"O extrator não devolveu os frames {faltam[:10]} de {blend.name}")
    return pd.DataFrame([linhas[f] for f in frames])
