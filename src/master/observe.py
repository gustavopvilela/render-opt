"""
Assina o fluxo de eventos do mestre e mostra cada evento.

python src/master/observe.py --mestre http://192.168.0.19:8000 --token SEGREDO
python src/master/observe.py ... --desde 0 --saida eventos.jsonl
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
import httpx

def assinar (url_mestre: str, token: Optional[str], desde: Optional[int] = None) -> Iterator[dict]:
    """
    Devolve os eventos um a um. Se a conexão cair, reconecta mandando `Last-Event-ID`:
    o mestre continua do último evento recebido, sem perder e repetir nenhum.
    """
    ultimo = desde
    while True:
        cabecalhos = {"X-Token": token} if token else {}
        if ultimo is not None:
            cabecalhos["Last-Event-ID"] = str(ultimo)
        try:
            with httpx.stream(
                "GET",
                f"{url_mestre}/eventos",
                headers=cabecalhos,
                timeout=httpx.Timeout(10.0, read=60.0)
            ) as resposta:
                resposta.raise_for_status()
                campos: dict[str, str] = {}
                for linha in resposta.iter_lines():
                    if linha == "": # Linha vazia fecha um evento
                        if "data" in campos:
                            ultimo = int(campos["id"])
                            yield json.loads(campos["data"])
                        campos = {}
                    elif not linha.startswith(":"): # : no início = comentário (ping)
                        nome, _, valor = linha.partition(":")
                        campos[nome] = valor.removeprefix(" ")
        except httpx.TransportError as erro:
            print(f"Conexão perdida ({type(erro).__name__}); reconectando em 3 s", file=sys.stderr)
            time.sleep(3)

def main (argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mostra os eventos do mestre.")
    parser.add_argument("--mestre", required=True, help="Ex.: http://192.168.0.19:8000")
    parser.add_argument("--token", default=None)
    parser.add_argument("--desde", type=int, default=None, help="0 = reenviar o histórico completo")
    parser.add_argument("--saida", type=Path, default=None, help="Arquivo de saída, grava um JSON por linha")
    args = parser.parse_args(argv)

    arquivo = args.saida.open("a", encoding="utf-8") if args.saida else None
    try:
        for ev in assinar(args.mestre.rstrip("/"), args.token, args.desde):
            hora = datetime.fromtimestamp(ev["t"]).strftime("%H:%M:%S")
            print(f"{hora} {ev['tipo']:<17} {ev['job_id'] or '':<16} {ev['maquina'] or '':<12} {ev['dados'] or ''}", flush=True)
            if arquivo:
                arquivo.write(json.dumps(ev, ensure_ascii=False) + "\n")
                arquivo.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if arquivo:
            arquivo.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())