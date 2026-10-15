"""
Submete um lote ao mestre. Roda na máquina do mestre.

python src/master/submit.py D:/cenas --lote L001 --config config/mestre.yaml
python src/master/submit.py D:/cenas --lote L002 --config config/mestre.yaml --distribuicao flamenco
"""

from __future__ import annotations
import argparse
import gzip
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path[0] = str(Path(__file__).resolve().parents[1])

import httpx2
import zstandard
from comum.util import sha256_arquivo, validar_id, versao_tupla
from master.config import carregar_config

SCRIPT_EMPACOTAR = Path(__file__).with_name("zip_scene.py")

def versao_do_arquivo (caminho: Path) -> str:
    """
    Lê no cabeçalho do .blend a versão que o gravou, sem abrir o Blender.
    """
    with caminho.open("rb") as arquivo:
        inicio = arquivo.read(4)
        arquivo.seek(0)
        if inicio[:2] == b"\x1f\x8b":
            cabecalho = gzip.GzipFile(fileobj=arquivo).read(17)
        elif inicio == b"\x28\xb5\x2f\xfd":
            cabecalho = zstandard.ZstdDecompressor().stream_reader(arquivo).read(17)
        else:
            cabecalho = arquivo.read(17)
    # Até o 4.x: BLENDER-v405; a partir do 5.0: BLENDER17-01v0501
    achado = re.match(rb"BLENDER(?:[_-][vV](\d)(\d\d)|\d\d-\d\d[vV](\d\d)(\d\d))", cabecalho)
    if achado is None:
        raise SystemExit(f"{caminho} não é um .blend")
    maior, menor = (int(g) for g in achado.groups() if g is not None)
    return f"{maior}.{menor}"

def escolher_blender (blenders: dict[str, str], versao: str) -> str:
    """
    Blender de menor versão que ainda abre a cena, assim a cópia mantém a série dela.
    """
    compativeis = sorted((v for v in blenders if versao_tupla(v) >= versao_tupla(versao)), key=versao_tupla)
    if not compativeis:
        raise SystemExit(f"Nenhum Blender do mestre abre uma cena {versao}")
    return blenders[compativeis[0]]

def empacotar (blender: str, origem: Path, destino: Path) -> tuple[str, list[int]]:
    """
    Roda zip_scene.py no Blender e devolve a versão e os frames da cena.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".parcial.blend")
    processo = subprocess.run(
        [
            blender,
            "--factory-startup",
            "-noaudio",
            "-b",
            str(origem),
            "-P",
            str(SCRIPT_EMPACOTAR),
            "--",
            str(temporario)
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    saida = processo.stdout + processo.stderr
    if processo.returncode != 0 or not temporario.is_file():
        print(saida[-3000:], file=sys.stderr)
        raise SystemExit(f"Falha ao empacotar {origem} (código {processo.returncode})")

    for linha in saida.splitlines():
        if linha.startswith(("RENDEROPT_FALTANDO", "RENDEROPT_BIBLIOTECA", "RENDEROPT_AVISO")):
            print("AVISO:", linha, file=sys.stderr)
        if "missing" in linha.lower() and "linked data-blocks" in linha:
            print("AVISO:", linha.strip(), file=sys.stderr)

    versao = re.search(r"RENDEROPT_VERSAO (\d+\.\d+)", saida)
    inicio, fim, passo = map(int, re.search(r"RENDEROPT_FRAMES (-?\d+) (-?\d+) (\d+)", saida).groups())
    temporario.replace(destino)
    return versao[1], list(range(inicio, fim + 1, passo))

def main (argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Submete um lote ao mestre do render-opt.")
    parser.add_argument("pasta", type=Path, help="Pasta com os .blend do lote")
    parser.add_argument("--lote", required=True, help="Identificador do lote, ex.: L001")
    parser.add_argument("--config", type=Path, required=True, help="Config do lote, ex.: config/mestre.yaml")
    parser.add_argument("--distribuicao", choices=["sa", "flamenco"], default="sa",
                        help="Plano do Simulated Annealing ou ordem de chegada do Flamenco")
    args = parser.parse_args(argv)

    cfg = carregar_config(args.config)
    lote_id = validar_id(args.lote)
    if not cfg.blender:
        raise SystemExit("Defina 'blender' em config/mestre.yaml para empacotar as cenas")
    arquivos = sorted(p for p in args.pasta.glob("*.blend") if not p.name.startswith("."))
    if not arquivos:
        raise SystemExit(f"Nenhum .blend em {args.pasta}")

    cenas = []
    for arquivo in arquivos:
        cena_id = validar_id(re.sub(r"[^A-Za-z0-9_.-]", "_", arquivo.stem))
        destino = cfg.pasta_cenas / f"{cena_id}.blend"
        original = versao_do_arquivo(arquivo)
        versao, frames = empacotar(escolher_blender(cfg.blender, original), arquivo, destino)
        print(f"{cena_id}: frames {frames[0]}-{frames[-1]} ({len(frames)}), gravada no {original}, empacotada no {versao}")
        cenas.append({
            "cena_id": cena_id,
            "cena_sha256": sha256_arquivo(destino),
            "versao_cena": versao,
            "frames": frames,
        })

    corpo = {"lote_id": lote_id, "cenas": cenas, "distribuicao": args.distribuicao}
    cabecalho = {"X-Token": cfg.token} if cfg.token else {}
    host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
    resposta = httpx2.post(
        f"http://{host}:{cfg.port}/lotes",
        json=corpo,
        headers=cabecalho,
        timeout=600
    )
    if resposta.status_code != 200:
        raise SystemExit(f"Mestre recusou o lote ({resposta.status_code}): {resposta.text}")
    print(resposta.json())
    return 0

if __name__ == "__main__":
    sys.exit(main())