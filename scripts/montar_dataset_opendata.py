#!/usr/bin/env python3
"""Monta o dataset supervisionado a partir do Blender Open Data + features extraidas.

Junta duas fontes que descrevem coisas diferentes:

  data/raw/<cena>.jsonl        features da cena (saida de src/extract/) -- 1 linha por frame
  data/raw/opendata-*.jsonl    medicoes de tempo por dispositivo        -- N linhas por cena

O dump do Open Data e aceito nas duas formas em que ele circula: o .jsonl puro
(como baixado do site, ~1,8 GB) ou o .zip que o contem. Em ambos os casos a
leitura e por streaming, linha a linha -- o arquivo nunca e carregado inteiro
na memoria.

O resultado e uma tabela CSV com uma linha por (cena, dispositivo, medicao),
carregando as features da cena, as features da maquina e os dois alvos de
regressao definidos em docs/alvo-do-modelo.md:

  vazao_amostras_pixel_s = amostras * pixels / time_for_samples   (modelo principal, secao 6)
  custo_fixo_s           = total_render_time - time_for_samples   (modelo secundario, secao 5)

Uso:
    python scripts/montar_dataset_opendata.py
    python scripts/montar_dataset_opendata.py --agregar mediana
    python scripts/montar_dataset_opendata.py --multi-gpu --saida data/processed/od_full.csv
    python scripts/montar_dataset_opendata.py --dump data/raw/opendata-2026-09-06-000000+0000.jsonl

Ver docs/alvo-do-modelo.md (secoes 5, 6 e 10.4) e docs/obtencao-de-dados.md (secao 6.3).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import io
import json
import math
import statistics as st
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent
DIR_FEATURES = RAIZ / "data" / "raw"
SAIDA_PADRAO = RAIZ / "data" / "processed" / "dataset_opendata.csv"

def descobrir_dump(dir_dados: Path = DIR_FEATURES) -> Path | None:
    """Acha o dump do Open Data em data/raw/, .jsonl ou .zip.

    O nome do arquivo baixado carrega a data do snapshot
    (opendata-2026-09-06-000000+0000.jsonl), entao nao da para fixar um nome
    padrao: procuramos por prefixo e ficamos com o mais recente.
    """
    candidatos = sorted(
        (p for p in dir_dados.glob("opendata-*")
         if p.suffix.lower() in (".jsonl", ".zip")),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    return candidatos[-1] if candidatos else None


# As 3 cenas do benchmark v4. O nome do arquivo <cena>.jsonl precisa bater
# com o scene.label do Open Data -- e por isso que o join funciona sem tabela
# de mapeamento.
CENAS_V4 = ("classroom", "junkshop", "monster")

# Colunas que o extrator le do arquivo .blend mas que o benchmark SOBRESCREVE
# em tempo de execucao. Mante-las seria envenenar o modelo com valores falsos:
# 'amostras' = 300 vem do .blend, mas o benchmark v4 renderiza por tempo e
# obtem entre 100 e 3000 amostras dependendo da maquina.
COLUNAS_CONTAMINADAS = frozenset({
    "amostras",                 # sobrescrito: v4 usa time_limit, nao sample count
    "amostras_x_pixels",        # derivado de 'amostras' -> errado
    "log_amostras_x_pixels",    # idem
    "cycles_device",            # sobrescrito por device_info.device_type
    "cycles_time_limit",        # sobrescrito por stats.time_limit = 30
    "threads_render",           # sobrescrito por device_info.num_cpu_threads
})

# Colunas que sao chave/identidade, nunca features (docs/alvo-do-modelo.md 10.1).
COLUNAS_CHAVE = ("origem", "cena", "arquivo_blend", "frame", "maquina")


# ---------------------------------------------------------------------------
# Lado A: features das cenas
# ---------------------------------------------------------------------------

def carregar_features(dir_features: Path, cenas=CENAS_V4) -> dict[str, dict]:
    """Le <cena>.jsonl e devolve {cena: linha_do_frame_1}.

    As cenas do benchmark sao imagens estaticas: existe um unico frame, o 1.
    Se o extrator emitiu varios frames (porque o .blend tem timeline), so o
    frame 1 corresponde ao que o benchmark mediu -- ver alvo-do-modelo.md 10.4.
    """
    features: dict[str, dict] = {}
    for cena in cenas:
        caminho = dir_features / f"{cena}.jsonl"
        if not caminho.exists():
            print(f"  [aviso] {caminho.name} nao encontrado -- cena ignorada", file=sys.stderr)
            continue

        linhas = [json.loads(l) for l in caminho.open(encoding="utf-8") if l.strip()]
        frame1 = [r for r in linhas if r.get("frame") == 1]
        if not frame1:
            print(f"  [aviso] {caminho.name} nao tem frame 1 -- cena ignorada", file=sys.stderr)
            continue

        linha = {k: v for k, v in frame1[0].items() if k not in COLUNAS_CONTAMINADAS}
        descartadas = sorted(set(frame1[0]) & COLUNAS_CONTAMINADAS)
        features[cena] = linha
        print(f"  {cena:12s} {len(linhas):4d} frames lidos, frame 1 mantido, "
              f"{len(descartadas)} colunas descartadas")
    return features


# ---------------------------------------------------------------------------
# Lado B: medicoes do Open Data
# ---------------------------------------------------------------------------

def _id_maquina(device_info: dict) -> tuple[str, str, int]:
    """Devolve (nome, tipo, n_dispositivos).

    device_type sozinho NAO identifica uma maquina (alvo-do-modelo.md 12.2):
    'OPTIX' abarca de uma RTX 2060 a uma RTX 5090. O nome do dispositivo e o
    identificador real.
    """
    comp = device_info.get("compute_devices") or []
    tipo = device_info.get("device_type", "?")
    if not comp:
        return ("?", tipo, 0)
    nomes = sorted(d.get("name", "?") for d in comp)
    return (" + ".join(nomes), tipo, len(comp))


def _features_maquina(elem: dict) -> dict:
    di = elem.get("device_info") or {}
    si = elem.get("system_info") or {}
    bv = elem.get("blender_version") or {}
    nome, tipo, n_dev = _id_maquina(di)

    versao = bv.get("version") or bv.get("label") or ""
    partes = versao.split(".")

    cpu = next((d.get("name") for d in (si.get("devices") or [])
                if d.get("type") == "CPU"), None)

    return {
        "maquina": f"{nome}|{tipo}",
        "dispositivo_nome": nome,
        "dispositivo_tipo": tipo,
        "num_dispositivos_computo": n_dev,
        "cpu_nome": cpu,
        "num_cpu_cores": si.get("num_cpu_cores"),
        "num_cpu_threads": di.get("num_cpu_threads") or si.get("num_cpu_threads"),
        "num_cpu_sockets": si.get("num_cpu_sockets"),
        "sistema": si.get("system"),
        "versao_blender_medicao": versao,
        "versao_blender_major": int(partes[0]) if partes and partes[0].isdigit() else None,
        "versao_blender_minor": int(partes[1]) if len(partes) > 1 and partes[1].isdigit() else None,
        "launcher": (elem.get("benchmark_launcher") or {}).get("label"),
    }


@contextlib.contextmanager
def abrir_dump(caminho: Path):
    """Abre o dump como um iteravel de linhas de texto, seja .jsonl ou .zip.

    Streaming nos dois casos: o dump tem ~1,8 GB descompactado e nao cabe
    confortavelmente na memoria.
    """
    if caminho.suffix.lower() == ".zip":
        z = zipfile.ZipFile(caminho)
        try:
            nome = next(n for n in z.namelist() if n.endswith(".jsonl"))
        except StopIteration:
            raise SystemExit(f"{caminho.name} nao contem nenhum .jsonl")
        print(f"  lendo {caminho.name} :: {nome}")
        with z.open(nome) as fh:
            yield io.TextIOWrapper(fh, "utf-8")
        z.close()
    else:
        tam = caminho.stat().st_size / 1e9
        print(f"  lendo {caminho.name} ({tam:.2f} GB)")
        with caminho.open(encoding="utf-8") as fh:
            yield fh


def iterar_medicoes_v4(caminho_dump: Path, cenas: set[str], manter_multi_gpu: bool):
    """Gera (cena, features_maquina, stats) para cada medicao v4 valida."""
    contagem = collections.Counter()

    with abrir_dump(caminho_dump) as fonte:
        for linha in fonte:
            if not linha.strip():
                continue
            reg = json.loads(linha)
            if reg.get("schema_version") != "v4":
                contagem["descartado: esquema != v4"] += 1
                continue

            for elem in reg.get("data") or []:
                cena = ((elem.get("scene") or {}).get("label") or "").lower()
                if cena not in cenas:
                    contagem["descartado: cena sem features"] += 1
                    continue

                stats = elem.get("stats") or {}
                n_amostras = stats.get("number_of_samples")
                t_amostras = stats.get("time_for_samples")
                t_total = stats.get("total_render_time")

                if not n_amostras or not t_amostras or t_amostras <= 0:
                    contagem["descartado: stats incompleto"] += 1
                    continue

                fm = _features_maquina(elem)

                # Vazao nao e aditiva entre dispositivos (alvo-do-modelo.md 12.5):
                # multi-GPU nao e a soma das GPUs. Por padrao fica de fora.
                if fm["num_dispositivos_computo"] != 1 and not manter_multi_gpu:
                    contagem["descartado: multi-GPU"] += 1
                    continue
                if fm["num_dispositivos_computo"] == 0:
                    contagem["descartado: sem dispositivo"] += 1
                    continue

                contagem["mantido"] += 1
                yield cena, fm, stats, t_total, t_amostras, n_amostras

    print("  contagem de registros:")
    for k, v in contagem.most_common():
        print(f"    {k:34s} {v:>9,d}".replace(",", "."))


# ---------------------------------------------------------------------------
# O join
# ---------------------------------------------------------------------------

def montar(caminho_dump: Path, dir_features: Path, manter_multi_gpu: bool) -> list[dict]:
    print("Lendo features das cenas:")
    features = carregar_features(dir_features)
    if not features:
        raise SystemExit("Nenhuma cena com features. Rode o extrator antes.")

    print("\nLendo medicoes do Open Data:")
    linhas: list[dict] = []
    negativos = 0

    for cena, fm, stats, t_total, t_amostras, n_amostras in iterar_medicoes_v4(
            caminho_dump, set(features), manter_multi_gpu):

        fc = features[cena]
        pixels = fc.get("pixels_totais")
        if not pixels:
            pixels = (fc.get("res_x") or 0) * (fc.get("res_y") or 0)
        if not pixels:
            continue

        # Alvo 1 -- vazao normalizada, em amostras*pixel por segundo.
        # Normalizar por pixels e o que impede o modelo de confundir mudanca
        # de resolucao com mudanca de dificuldade de cena (alvo-do-modelo.md 6.2).
        trabalho = n_amostras * pixels
        vazao = trabalho / t_amostras

        # Alvo 2 -- custo fixo: BVH, compilacao de kernel, upload de textura.
        # So o v4 permite medi-lo, porque grava dois relogios (alvo-do-modelo.md 5).
        custo_fixo = None
        if t_total is not None:
            custo_fixo = t_total - t_amostras
            if custo_fixo < 0:          # ruido de arredondamento em medicoes rapidas
                negativos += 1
                custo_fixo = None

        linha = {
            # --- chaves (nunca features; ver alvo-do-modelo.md 10.1)
            "origem": "opendata",
            "cena": cena,
            "arquivo_blend": fc.get("arquivo_blend"),
            "frame": 1,                 # cenas do benchmark sao stills
            **fm,                       # inclui 'maquina'
            # --- alvos
            "vazao_amostras_pixel_s": vazao,
            "log_vazao": math.log(vazao),
            "custo_fixo_s": custo_fixo,
            "log_custo_fixo": math.log(custo_fixo) if custo_fixo and custo_fixo > 0 else None,
            # --- contexto da medicao (diagnostico, NAO features do modelo de vazao)
            "medido_num_amostras": n_amostras,
            "medido_time_for_samples": t_amostras,
            "medido_total_render_time": t_total,
            "medido_samples_per_minute": stats.get("samples_per_minute"),
            "medido_pico_memoria_mb": stats.get("device_peak_memory"),
            "pixels_totais": pixels,
        }
        # features da cena por ultimo, sem sobrescrever o que ja esta acima
        for k, v in fc.items():
            linha.setdefault(k, v)

        linhas.append(linha)

    if negativos:
        print(f"  [aviso] {negativos} medicoes com custo fixo negativo -> custo_fixo_s vazio")
    return linhas


def agregar_mediana(linhas: list[dict], min_medicoes: int = 1) -> list[dict]:
    """Colapsa para uma linha por (maquina, cena), usando mediana dos alvos.

    Util para o experimento de generalizacao por dispositivo: reduz ~700 mil
    medicoes a ~2.600 linhas (870 dispositivos x 3 cenas) e remove o peso
    desproporcional de placas populares, que submetem milhares de vezes.
    """
    grupos: dict[tuple, list[dict]] = collections.defaultdict(list)
    for l in linhas:
        grupos[(l["maquina"], l["cena"])].append(l)

    saida = []
    for (_maquina, _cena), g in grupos.items():
        if len(g) < min_medicoes:
            continue
        base = dict(g[0])
        base["n_medicoes"] = len(g)
        for campo in ("vazao_amostras_pixel_s", "custo_fixo_s",
                      "medido_num_amostras", "medido_pico_memoria_mb"):
            vals = [x[campo] for x in g if x.get(campo) is not None]
            base[campo] = st.median(vals) if vals else None
        base["log_vazao"] = math.log(base["vazao_amostras_pixel_s"])
        cf = base["custo_fixo_s"]
        base["log_custo_fixo"] = math.log(cf) if cf and cf > 0 else None
        for campo in ("medido_time_for_samples", "medido_total_render_time",
                      "medido_samples_per_minute"):
            base.pop(campo, None)
        saida.append(base)
    return saida


def escrever_csv(linhas: list[dict], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    # ordem estavel: chaves primeiro, depois o resto em ordem de aparicao
    colunas = list(COLUNAS_CHAVE)
    for l in linhas:
        for k in l:
            if k not in colunas:
                colunas.append(k)
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=colunas, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)


def resumo(linhas: list[dict]) -> None:
    print(f"\n{'=' * 62}")
    print(f"linhas: {len(linhas):,d}".replace(",", "."))
    maquinas = {l["maquina"] for l in linhas}
    print(f"maquinas distintas: {len(maquinas):,d}".replace(",", "."))

    por_cena = collections.Counter(l["cena"] for l in linhas)
    print("\nlinhas por cena:")
    for c, n in por_cena.most_common():
        print(f"  {c:12s} {n:>9,d}".replace(",", "."))

    por_tipo = collections.Counter(l["dispositivo_tipo"] for l in linhas)
    print("\nlinhas por tipo de dispositivo:")
    for t, n in por_tipo.most_common():
        print(f"  {t:12s} {n:>9,d}".replace(",", "."))

    print("\nmediana dos alvos por cena:")
    print(f"  {'cena':12s} {'vazao (Mamostra*px/s)':>22s} {'custo fixo (s)':>16s}")
    for c in sorted(por_cena):
        v = [l["vazao_amostras_pixel_s"] for l in linhas if l["cena"] == c]
        cf = [l["custo_fixo_s"] for l in linhas
              if l["cena"] == c and l.get("custo_fixo_s") is not None]
        mv = st.median(v) / 1e6
        mc = st.median(cf) if cf else float("nan")
        print(f"  {c:12s} {mv:22.1f} {mc:16.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", "--zip", type=Path, default=None, dest="dump",
                    help="dump do Open Data (.jsonl ou .zip); por padrao pega o "
                         "opendata-* mais recente de data/raw/")
    ap.add_argument("--features", type=Path, default=DIR_FEATURES)
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    ap.add_argument("--multi-gpu", action="store_true",
                    help="mantem submissoes com mais de um dispositivo de computo "
                         "(por padrao sao descartadas -- vazao nao e aditiva)")
    ap.add_argument("--agregar", choices=("nenhum", "mediana"), default="nenhum",
                    help="colapsa para uma linha por (maquina, cena)")
    ap.add_argument("--min-medicoes", type=int, default=1,
                    help="com --agregar mediana, exige ao menos N submissoes por "
                         "(maquina, cena); 30 reproduz o recorte da secao 6.4.1")
    args = ap.parse_args()

    dump = args.dump or descobrir_dump(args.features)
    if dump is None:
        raise SystemExit(
            f"Nao encontrei nenhum opendata-*.jsonl ou opendata-*.zip em "
            f"{args.features}. Baixe o dump do Open Data primeiro.")
    if not dump.exists():
        raise SystemExit(f"Nao encontrei {dump}.")

    linhas = montar(dump, args.features, args.multi_gpu)
    if not linhas:
        raise SystemExit("Nenhuma linha produzida.")

    if args.agregar == "mediana":
        antes = len(linhas)
        linhas = agregar_mediana(linhas, args.min_medicoes)
        print(f"\nagregacao por mediana: {antes:,d} -> {len(linhas):,d} linhas"
              .replace(",", "."))

    resumo(linhas)
    escrever_csv(linhas, args.saida)
    print(f"\nescrito: {args.saida}")


if __name__ == "__main__":
    main()
