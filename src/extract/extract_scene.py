"""
Entrypoint executado DENTRO do Blender.

    blender -b cena.blend --factory-startup -noaudio \
        -P src/renderopt/extract/extract_scene.py -- --saida data/raw/cena.jsonl

Este script apenas LE a cena: nao altera configuracoes de render e nao
renderiza nada. Qualquer mutacao invalidaria o benchmark, ja que as features
extraidas precisam descrever exatamente a cena que sera cronometrada.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import bpy

# Permite `import extract.*` quando o script e invocado pelo Blender, que nao
# conhece o venv nem a raiz do projeto. `parents[1]` e a pasta `src/`.
_RAIZ_SRC = Path(__file__).resolve().parents[1]
if str(_RAIZ_SRC) not in sys.path:
    sys.path.insert(0, str(_RAIZ_SRC))

from extract import features  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Le apenas os argumentos apos `--`, que o Blender repassa ao script."""
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(prog="extract_scene")
    parser.add_argument("--saida", required=True, help="arquivo .jsonl de saida")
    parser.add_argument(
        "--frames",
        default=None,
        help="lista/intervalo de frames, ex.: '1,5,10' ou '1-250'",
    )
    parser.add_argument(
        "--passo", type=int, default=1, help="amostra 1 a cada N frames da timeline"
    )
    parser.add_argument(
        "--todas-cenas",
        action="store_true",
        help="extrai todas as cenas do arquivo, nao apenas a ativa",
    )
    return parser.parse_args(argv)


def resolver_frames(spec: str | None, scene: bpy.types.Scene) -> list[int] | None:
    if not spec:
        return None
    frames: list[int] = []
    for parte in spec.split(","):
        parte = parte.strip()
        if "-" in parte:
            inicio, fim = parte.split("-", 1)
            frames.extend(range(int(inicio), int(fim) + 1))
        elif parte:
            frames.append(int(parte))
    return frames


def main() -> int:
    args = parse_args(list(sys.argv))
    cenas = list(bpy.data.scenes) if args.todas_cenas else [bpy.context.scene]

    linhas: list[dict[str, Any]] = []
    for scene in cenas:
        frames = resolver_frames(args.frames, scene)
        linhas.extend(features.extrair_cena(scene, frames=frames, passo=args.passo))

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("w", encoding="utf-8") as arquivo:
        for linha in linhas:
            arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")

    # Prefixo reconhecido pelo orquestrador ao parsear o stdout do Blender.
    print(f"[RENDEROPT] ok linhas={len(linhas)} saida={saida}")
    return 0


if __name__ == "__main__":
    codigo = main()
    # Sem isto o Blender ignora o codigo de saida e sempre retorna 0.
    if codigo != 0:
        os._exit(codigo)
