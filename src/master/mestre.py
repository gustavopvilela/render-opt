from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path[0] = str(Path(__file__).resolve().parents[1])

import uvicorn
from master.app import criar_app
from master.config import carregar_config

def main (argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="No mestre do render-opt.")
    parser.add_argument("--config", type=Path, required=True, help="Arquivo YAML de configuração.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = carregar_config(args.config)
    uvicorn.run(criar_app(cfg), host=cfg.host, port=cfg.port, timeout_graceful_shutdown=5)
    return 0

if __name__ == "__main__":
    sys.exit(main())