"""
Exporta o plano e o que foi executado num lote, em CSV, para os gráficos. Roda na máquina do mestre.

python src/master/exportar.py L001 --config config/mestre.yaml
"""

from __future__ import annotations
import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

sys.path[0] = str(Path(__file__).resolve().parents[1])

from master.config import carregar_config

FIM_DA_TENTATIVA = {"concluido", "adiado", "devolvido"}

CAMPOS_PLANO = ["job_id", "lote_id", "maquina", "posicao", "inicio_s", "setup_s", "processamento_s", "fim_s", "cena_id", "frame"]
CAMPOS_EXECUCAO = ["job_id", "cena_id", "frame", "tentativa", "maquina", "desfecho", "inicio_s", "fim_s", "duracao_mestre_s",
                   "tempo_trabalhador_s", "preparacao_s", "amostragem_s", "gravacao_s", "previsto_s", "maquina_planejada",
                   "posicao_planejada", "inicio_planejado_s", "fim_planejado_s", "motivo"]
CAMPOS_EVENTOS = ["t_s", "tipo", "job_id", "maquina", "dados"]

def escrever_csv (caminho: Path, campos: list[str], linhas: list[dict]) -> None:
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)

def exportar (pasta_dados: Path, lote_id: str, saida: Path) -> dict:
    conexao = sqlite3.connect(f"file:{pasta_dados / 'estado.sqlite3'}?mode=ro", uri=True)
    conexao.row_factory = sqlite3.Row
    lote = conexao.execute("SELECT * FROM lotes WHERE id = ?", (lote_id,)).fetchone()
    if lote is None:
        raise SystemExit(f"Lote {lote_id} não existe")
    pasta_lote = pasta_dados / "resultados" / lote_id
    t0 = lote["inicio"] if lote["inicio"] is not None else lote["criado_em"]

    jobs = {j["id"]: j for j in conexao.execute("SELECT * FROM jobs WHERE lote_id = ?", (lote_id,))}
    plano = {}
    if conexao.execute("SELECT 1 FROM sqlite_master WHERE name = 'plano'").fetchone():
        plano = {p["job_id"]: dict(p) for p in conexao.execute("SELECT * FROM plano WHERE lote_id = ? ORDER BY maquina, posicao", (lote_id,))}
    arquivo_plano = pasta_lote / "plano.json"
    estimativas = {}
    if arquivo_plano.is_file():
        estimativas = {j["job_id"]: j["estimativas_s"] for j in json.loads(arquivo_plano.read_text(encoding="utf-8"))["jobs"]}

    eventos = [e for e in conexao.execute("SELECT * FROM eventos WHERE t >= ? AND t <= ? ORDER BY id",
                                          (lote["criado_em"], lote["fim"] if lote["fim"] is not None else float("inf")))
               if e["job_id"] in jobs or (e["job_id"] is None and e["tipo"] == "trabalhador_caiu")]

    abertas, tentativas, contagem = {}, [], {}
    for e in eventos:
        if e["job_id"] is None:
            continue
        if e["tipo"] == "entregue":
            contagem[e["job_id"]] = contagem.get(e["job_id"], 0) + 1
            abertas[e["job_id"]] = (e, contagem[e["job_id"]])
        elif e["tipo"] in FIM_DA_TENTATIVA and e["job_id"] in abertas:
            inicio, n = abertas.pop(e["job_id"])
            tentativas.append((inicio, e, n))
    tentativas += [(inicio, None, n) for inicio, n in abertas.values()]

    linhas_execucao = []
    for inicio, fim, n in sorted(tentativas, key=lambda x: x[0]["id"]):
        job = jobs[inicio["job_id"]]
        planejado = plano.get(job["id"], {})
        relatorio = {}
        candidatos = list((pasta_lote / "relatorios").glob(f"{job['id']}_{(inicio['dados'] or '')[:8]}*.json"))
        if len(candidatos) == 1:
            relatorio = json.loads(candidatos[0].read_text(encoding="utf-8"))
        linhas_execucao.append({
            "job_id": job["id"],
            "cena_id": job["cena_id"],
            "frame": job["frame"],
            "tentativa": n,
            "maquina": inicio["maquina"],
            "desfecho": fim["tipo"] if fim else "em_andamento",
            "inicio_s": inicio["t"] - t0,
            "fim_s": fim["t"] - t0 if fim else None,
            "duracao_mestre_s": fim["t"] - inicio["t"] if fim else None,
            "tempo_trabalhador_s": relatorio.get("tempo_total_s"),
            "preparacao_s": relatorio.get("tempo_preparacao_s"),
            "amostragem_s": relatorio.get("tempo_amostragem_s"),
            "gravacao_s": relatorio.get("tempo_gravacao_s"),
            "previsto_s": estimativas.get(job["id"], {}).get(inicio["maquina"]),
            "maquina_planejada": planejado.get("maquina"),
            "posicao_planejada": planejado.get("posicao"),
            "inicio_planejado_s": planejado.get("inicio_s"),
            "fim_planejado_s": planejado.get("fim_s"),
            "motivo": relatorio.get("motivo"),
        })

    linhas_plano = [{**p, "cena_id": jobs[p["job_id"]]["cena_id"], "frame": jobs[p["job_id"]]["frame"]} for p in plano.values()]
    linhas_eventos = [{"t_s": e["t"] - t0, "tipo": e["tipo"], "job_id": e["job_id"], "maquina": e["maquina"], "dados": e["dados"]}
                      for e in eventos]

    maquinas = sorted({p["maquina"] for p in plano.values()} | {l["maquina"] for l in linhas_execucao})
    por_maquina = {}
    for m in maquinas:
        planejados = [p for p in plano.values() if p["maquina"] == m]
        executadas = [l for l in linhas_execucao if l["maquina"] == m and l["duracao_mestre_s"] is not None]
        por_maquina[m] = {
            "frames_planejados": len(planejados),
            "frames_concluidos": sum(l["desfecho"] == "concluido" for l in executadas),
            "concluidos_fora_do_plano": sum(l["desfecho"] == "concluido" and l["maquina_planejada"] != m for l in executadas),
            "fim_planejado_s": max((p["fim_s"] for p in planejados), default=0.0),
            "fim_real_s": max((l["fim_s"] for l in executadas), default=0.0),
            "ocupado_s": sum(l["duracao_mestre_s"] for l in executadas),
            "perdido_s": sum(l["duracao_mestre_s"] for l in executadas if l["desfecho"] != "concluido"),
        }
    resumo = {
        "lote_id": lote_id,
        "distribuicao": lote["distribuicao"],
        "makespan_planejado_s": max((p["fim_s"] for p in plano.values()), default=None),
        "makespan_real_s": lote["fim"] - lote["inicio"] if lote["fim"] is not None and lote["inicio"] is not None else None,
        "eventos": {t: sum(e["tipo"] == t for e in eventos) for t in ("roubo", "adiado", "devolvido", "orfao", "trabalhador_caiu", "falha_definitiva")},
        "por_maquina": por_maquina,
    }

    saida.mkdir(parents=True, exist_ok=True)
    escrever_csv(saida / "plano.csv", CAMPOS_PLANO, linhas_plano)
    escrever_csv(saida / "execucao.csv", CAMPOS_EXECUCAO, linhas_execucao)
    escrever_csv(saida / "eventos.csv", CAMPOS_EVENTOS, linhas_eventos)
    (saida / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo

def main (argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Exporta o plano e a execução de um lote do render-opt.")
    parser.add_argument("lote", help="Identificador do lote, ex.: L001")
    parser.add_argument("--config", type=Path, required=True, help="Config do mestre, ex.: config/mestre.yaml")
    parser.add_argument("--saida", type=Path, help="Pasta dos CSV; padrão: resultados/<lote>/analise")
    args = parser.parse_args(argv)
    config = carregar_config(args.config)
    saida = args.saida or config.pasta_resultados / args.lote / "analise"
    resumo = exportar(config.pasta_dados, args.lote, saida)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"CSV em {saida}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
