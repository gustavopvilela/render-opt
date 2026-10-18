from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from comum.util import sha256_arquivo
from predict import esquema, features_cena
from predict.hardware import linha_maquina

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "data" / "processed" / "producao.parquet"
CAMPOS = ("job_id", "trabalhador_id", "status", "cena_id", "frame", "dispositivo", "sistema", "versao_blender",
          "tempo_previsto_s", "tempo_total_s", "tempo_preparacao_s", "tempo_blender_s",
          "amostra_final", "amostras_total", "instante_ultima_amostra_s")


def ler_relatorios (pasta_resultados: Path) -> pd.DataFrame:
    linhas = []
    for arq in sorted(pasta_resultados.glob("*/relatorios/*.json")):
        r = json.loads(arq.read_text(encoding="utf-8"))
        linhas.append({**{c: r.get(c) for c in CAMPOS}, "lote_id": arq.parent.parent.name})
    return pd.DataFrame(linhas, columns=list(CAMPOS) + ["lote_id"])


def ler_banco (caminho_db: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(f"file:{caminho_db.as_posix()}?mode=ro", uri=True) as con:
        jobs = pd.read_sql_query("SELECT id AS job_id, cena_sha256 FROM jobs", con)
        trab = pd.read_sql_query("SELECT id AS trabalhador_id, hardware FROM trabalhadores", con)
    trab["hardware"] = trab["hardware"].map(json.loads)
    return jobs, trab


def separar_tempos (rel: pd.DataFrame) -> pd.DataFrame:
    d = rel.copy()
    num = ["tempo_total_s", "tempo_preparacao_s", "tempo_blender_s", "instante_ultima_amostra_s",
           "amostra_final", "amostras_total"]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d["amostragem_s"] = d["instante_ultima_amostra_s"] - d["tempo_preparacao_s"]
    d["carregar_s"] = d["tempo_total_s"] - d["tempo_blender_s"]
    d["preparar_finalizar_s"] = d["tempo_blender_s"] - d["amostragem_s"]
    regras = [
        ("abortado", d["status"].astype(str).str.lower() == "abortado"),
        ("falha", d["status"].astype(str).str.lower() != "concluido"),
        ("amostragem incompleta", d["amostra_final"] != d["amostras_total"]),
        ("sem relógio do log", d[num].isna().any(axis=1)),
        ("tempo impossível", (d["amostragem_s"] <= 0) | (d["carregar_s"] <= 0) | (d["preparar_finalizar_s"] <= 0)),
    ]
    d["descarte"] = ""
    for motivo, cond in regras:
        d.loc[(d["descarte"] == "") & cond.fillna(True), "descarte"] = motivo
    return d


def atributos_dos_frames (d: pd.DataFrame, pasta_dados: Path, blender: str | None) -> pd.DataFrame:
    partes = []
    for (sha, cena_id), g in d.groupby(["cena_sha256", "cena_id"]):
        frames = sorted(set(int(f) for f in g["frame"]))
        cache = pasta_dados / "features" / f"{sha}.jsonl"
        linhas = {}
        if cache.exists():
            for texto in cache.read_text(encoding="utf-8").splitlines():
                if texto.strip():
                    r = json.loads(texto)
                    linhas[int(r["frame"])] = r
        faltam = [f for f in frames if f not in linhas]
        blend = pasta_dados / "cenas" / f"{cena_id}.blend"
        if faltam and blender and blend.exists() and sha256_arquivo(blend) == sha:
            extra = features_cena.extrair(blender, blend, faltam, RAIZ / "data" / "raw" / "cache_atributos")
            linhas.update({int(r["frame"]): r for r in extra.to_dict("records")})
        elif faltam:
            print(f"[aviso] {cena_id} ({sha[:12]}): {len(faltam)} frames sem atributos no cache; ficam de fora")
        for f in frames:
            if f in linhas:
                partes.append({"cena_sha256": sha, "frame": f,
                               **{c: linhas[f].get(c) for c in esquema.COLUNAS_CENA}})
    return pd.DataFrame(partes)


def montar (pasta_dados: Path, blender: str | None) -> pd.DataFrame:
    rel = ler_relatorios(pasta_dados / "resultados")
    if rel.empty:
        raise SystemExit(f"Nenhum relatório em {pasta_dados / 'resultados'}")
    jobs, trab = ler_banco(pasta_dados / "estado.sqlite3")
    d = separar_tempos(rel).merge(jobs, on="job_id", how="left")
    d.loc[(d["descarte"] == "") & d["cena_sha256"].isna(), "descarte"] = "job fora do banco"
    print(f"relatórios: {len(d):,}")
    print("descartados:", d.loc[d["descarte"] != "", "descarte"].value_counts().to_dict())
    d = d[d["descarte"] == ""].copy()
    d["frame"] = d["frame"].astype(int)

    cena = atributos_dos_frames(d, pasta_dados, blender)
    d = d.merge(cena, on=["cena_sha256", "frame"], how="inner")

    fichas = dict(zip(trab["trabalhador_id"], trab["hardware"]))
    maq = pd.DataFrame([linha_maquina(r.dispositivo, fichas.get(r.trabalhador_id, {}), r.versao_blender, r.trabalhador_id)
                        for r in d.itertuples()], index=d.index)
    d = pd.concat([d.drop(columns=[c for c in maq.columns if c in d.columns]), maq], axis=1)

    d["vazao_amostras_pixel_s"] = d["amostras_total"] * pd.to_numeric(d["pixels_totais"]) / d["amostragem_s"]
    d["log_vazao"] = np.log(d["vazao_amostras_pixel_s"])
    d["log_carregar"] = np.log(d["carregar_s"])
    d["log_preparar"] = np.log(d["preparar_finalizar_s"])
    d["cena"] = d["cena_id"]
    d["fonte"] = "producao"
    return d.drop(columns=["status", "descarte", "cena_id", "dispositivo"]).reset_index(drop=True)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta-dados", type=Path, required=True, help="a pasta_dados do mestre.yaml")
    ap.add_argument("--blender", default=None, help="Blender para extrair frames que não estão no cache")
    ap.add_argument("--saida", type=Path, default=SAIDA)
    args = ap.parse_args()

    d = montar(args.pasta_dados, args.blender)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(args.saida, index=False)

    print(f"\naproveitados: {len(d):,} relatórios, {d['cena_sha256'].nunique()} cenas, "
          f"{d['trabalhador_id'].nunique()} trabalhadores, {d['lote_id'].nunique()} lotes")
    print(d.groupby("trabalhador_id")[["carregar_s", "preparar_finalizar_s", "amostragem_s", "tempo_total_s"]]
          .median().round(2).to_string())
    r = np.log(d["tempo_previsto_s"] / d["tempo_total_s"])
    print(f"\nprevisto ÷ real (modelo em uso): fator mediano {np.exp(np.median(np.abs(r))):.3f}, "
          f"viés mediano {np.median(r):+.3f} em log")
    print(f"escrito: {args.saida}")


if __name__ == "__main__":
    main()
