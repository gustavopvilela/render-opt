"""
Orquestrador executado FORA do Blender (no venv do projeto).

Percorre um diretorio de arquivos .blend, dispara um processo Blender isolado
por arquivo e consolida as linhas em um unico dataset.

    python -m renderopt.extract.run_batch data/raw/cenas --saida data/processed/features.csv

Isolar cada arquivo em seu proprio processo e proposital: um .blend corrompido,
um add-on ausente ou um estouro de memoria derruba somente aquela extracao, e o
lote continua. Tambem permite apontar um binario Blender diferente por arquivo,
o que e o mecanismo de suporte a versoes 4.x e 5.x.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

TIMEOUT_PADRAO = 900  # s; cenas pesadas levam minutos so para abrir

# Assinatura do cabecalho de um arquivo .blend (offset 0): "BLENDER" + ponteiro
# + endianness + 3 digitos de versao. Ex.: b"BLENDER-v405", b"BLENDER-v500".
_MAGIC = b"BLENDER"


def versao_do_blend(caminho: Path) -> Optional[tuple[int, int]]:
    """
    Le a versao do Blender que gravou o arquivo sem abrir o Blender.

    Suporta arquivos comprimidos (Zstandard, padrao desde a 3.0), nos quais o
    cabecalho nao esta em texto puro -- nesse caso devolve None e o chamador
    recorre ao binario padrao.
    """
    with caminho.open("rb") as arquivo:
        cabecalho = arquivo.read(12)
    if not cabecalho.startswith(_MAGIC):
        return None  # provavelmente comprimido
    digitos = cabecalho[9:12].decode("ascii", errors="ignore")
    if not digitos.isdigit():
        return None
    return int(digitos[0]), int(digitos[1:])


def escolher_binario(
    versao: Optional[tuple[int, int]], binarios: dict[str, str], padrao: str
) -> str:
    """
    Seleciona o executavel do Blender compativel com o arquivo.

    Regra: um arquivo salvo na versao X.Y abre em qualquer Blender >= X.Y
    (compatibilidade para tras e garantida), mas um .blend 5.x NAO abre em um
    Blender 4.x. Escolhe-se, portanto, o menor binario disponivel cuja versao
    seja >= a do arquivo; na ausencia de correspondencia, usa-se o mais novo.
    """
    if versao is None or not binarios:
        return padrao

    candidatos = sorted(
        (tuple(int(p) for p in chave.split(".")), caminho)
        for chave, caminho in binarios.items()
    )
    for chave, caminho in candidatos:
        if chave >= versao:
            return caminho
    return candidatos[-1][1] if candidatos else padrao


def extrair_arquivo(
    blend: Path,
    saida_jsonl: Path,
    binario: str,
    script: Path,
    passo: int,
    frames: Optional[str],
    timeout: int,
) -> tuple[bool, str]:
    comando = [
        binario,
        "-b",
        str(blend),
        "--factory-startup",
        "-noaudio",
        "-P",
        str(script),
        "--",
        "--saida",
        str(saida_jsonl),
        "--passo",
        str(passo),
    ]
    if frames:
        comando += ["--frames", frames]

    try:
        proc = subprocess.run(
            comando, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout apos {timeout}s"

    if "[RENDEROPT] ok" not in proc.stdout:
        erro = (proc.stderr or proc.stdout).strip().splitlines()
        return False, erro[-1] if erro else "falha desconhecida"
    return True, ""


def consolidar(jsonls: Iterable[Path], saida: Path) -> int:
    """Une os .jsonl em um CSV (ou Parquet, se pandas estiver disponivel)."""
    linhas: list[dict[str, Any]] = []
    for caminho in jsonls:
        if not caminho.exists():
            continue
        with caminho.open(encoding="utf-8") as arquivo:
            linhas.extend(json.loads(linha) for linha in arquivo if linha.strip())

    if not linhas:
        return 0

    saida.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        df = pd.DataFrame(linhas)
        if saida.suffix == ".parquet":
            df.to_parquet(saida, index=False)
        else:
            df.to_csv(saida, index=False)
    except ImportError:
        import csv

        colunas = sorted({chave for linha in linhas for chave in linha})
        with saida.open("w", encoding="utf-8", newline="") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=colunas)
            escritor.writeheader()
            escritor.writerows(linhas)
    return len(linhas)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="run_batch")
    parser.add_argument("entrada", type=Path, help="diretorio com arquivos .blend")
    parser.add_argument("--saida", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--temp", type=Path, default=Path("data/raw/jsonl"))
    parser.add_argument("--passo", type=int, default=1)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_PADRAO)
    parser.add_argument(
        "--blender",
        default=os.environ.get("BLENDER_BIN") or shutil.which("blender") or "blender",
        help="binario padrao do Blender",
    )
    parser.add_argument(
        "--blender-versao",
        action="append",
        default=[],
        metavar="X.Y=CAMINHO",
        help="binario adicional por versao; repetir. Ex.: --blender-versao 4.2=/opt/blender42/blender",
    )
    args = parser.parse_args(argv)

    binarios: dict[str, str] = {}
    for entrada in args.blender_versao:
        chave, _, caminho = entrada.partition("=")
        binarios[chave.strip()] = caminho.strip()

    script = Path(__file__).with_name("extract_scene.py")
    args.temp.mkdir(parents=True, exist_ok=True)

    blends = sorted(args.entrada.rglob("*.blend"))
    if not blends:
        print(f"nenhum .blend encontrado em {args.entrada}", file=sys.stderr)
        return 1

    gerados: list[Path] = []
    falhas: list[tuple[Path, str]] = []
    for indice, blend in enumerate(blends, start=1):
        versao = versao_do_blend(blend)
        binario = escolher_binario(versao, binarios, args.blender)
        rotulo = f"{versao[0]}.{versao[1]}" if versao else "?"
        print(f"[{indice}/{len(blends)}] {blend.name} (blend {rotulo})", flush=True)

        jsonl = args.temp / f"{blend.stem}.jsonl"
        ok, erro = extrair_arquivo(
            blend, jsonl, binario, script, args.passo, args.frames, args.timeout
        )
        if ok:
            gerados.append(jsonl)
        else:
            falhas.append((blend, erro))
            print(f"    FALHA: {erro}", file=sys.stderr)

    total = consolidar(gerados, args.saida)
    print(f"\n{total} linhas em {args.saida}")
    if falhas:
        print(f"{len(falhas)} arquivo(s) falharam:", file=sys.stderr)
        for blend, erro in falhas:
            print(f"  - {blend}: {erro}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
