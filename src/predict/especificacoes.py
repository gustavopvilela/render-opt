from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PASTA = RAIZ / "data" / "raw" / "especificacoes"
SAIDA = RAIZ / "data" / "processed" / "especificacoes.parquet"

DISPOSITIVO = ("esp_lancamento", "esp_tdp_w", "esp_processo_nm", "esp_clock_ghz", "esp_unidades",
               "esp_cache_mb", "esp_fp32_tflops", "esp_banda_gbs", "esp_memoria_gb", "esp_rt_cores",
               "esp_passmark", "esp_arquitetura")
HOSPEDEIRA = ("host_passmark", "host_clock_ghz", "host_lancamento")
COLUNAS = DISPOSITIVO + HOSPEDEIRA

_PARCEIRAS = (r"(EVGA|ASUS|MSI|GIGABYTE|ZOTAC|XFX|SAPPHIRE|POWERCOLOR|ASROCK|PALIT|PNY|GAINWARD|INNO3D|"
              r"COLORFUL|GALAX|KFA2|DELL|HP|LENOVO|ACER|BIOSTAR|YESTON|SPARKLE|ELSA)")


def chave_cpu (nome: Optional[str]) -> str:
    n = re.sub(r"\((R|TM)\)|®|™", "", str(nome or ""))
    n = re.sub(r"@.*$", "", n)
    n = re.sub(r"\b\d{1,2}(ST|ND|RD|TH)\s+GEN\b", "", n, flags=re.I)
    n = re.sub(r"\b(AMD|INTEL|PROCESSOR|CPU|WITH RADEON GRAPHICS|W/ RADEON \S+ GRAPHICS)\b", "", n, flags=re.I)
    n = re.sub(r"\b\d+-CORES?\b|\b(DUAL|TRIPLE|QUAD|SIX|EIGHT|TEN|TWELVE|SIXTEEN)-CORE\b", "", n, flags=re.I)
    n = re.sub(r"\s0\s*$", "", n)
    return re.sub(r"[^a-z0-9]", "", n.lower())


def chave_gpu (nome: Optional[str]) -> str:
    n = re.sub(r"\((R|TM)\)|®|™", "", str(nome or ""))
    partes = n.split(" + ")
    n = next((p for p in partes if re.search(r"NVIDIA|GEFORCE|RADEON RX|RADEON PRO|\bARC", p, re.I)), partes[-1])
    n = re.sub(r"\s+(DG2|DG1)$", "", n.strip(), flags=re.I)
    n = re.sub(r"^" + _PARCEIRAS + r"\s+", "", n.strip(), flags=re.I)
    n = re.sub(r"LAPTOP GPU", "Mobile", n, flags=re.I)
    n = re.sub(r"WITH MAX-Q DESIGN", "Max-Q", n, flags=re.I)
    n = re.sub(r"^(NVIDIA|AMD|ATI|INTEL|APPLE)\s+", "", n.strip(), flags=re.I)
    n = re.sub(r"\bGRAPHICS\b", "", n, flags=re.I)
    n = re.sub(r"\[.*?\]|\(.*?\)", "", n)
    return re.sub(r"[^a-z0-9]", "", n.lower())


def _ano (data) -> float:
    m = re.match(r"(\d{4})(?:-(\d{2}))?", str(data or ""))
    return float(m[1]) + (int(m[2]) - 1) / 12 if m and m[2] else float(m[1]) if m else np.nan


def _num (v) -> float:
    m = re.search(r"\d+(?:\.\d+)?", str(v if v is not None else ""))
    return float(m[0]) if m else np.nan


def tabela_cpu (pasta: Path = PASTA) -> pd.DataFrame:
    tpu = pd.read_csv(pasta / "techpowerup_cpu.csv", encoding="utf-8-sig")
    a = pd.DataFrame({
        "chave": (tpu["brand"] + " " + tpu["model"]).map(chave_cpu),
        "nome_base": tpu["model"], "fonte": "TechPowerUp",
        "esp_lancamento": tpu["released_date"].map(_ano), "esp_tdp_w": tpu["tdp_w"].map(_num),
        "esp_processo_nm": tpu["process_nm"].map(_num),
        "esp_clock_ghz": tpu["boost_clock_ghz"].map(_num).fillna(tpu["base_clock_ghz"].map(_num)),
        "esp_unidades": tpu["cores"].map(_num), "esp_cache_mb": tpu["l3_cache_mb"].map(_num),
        "esp_passmark": tpu["passmark_cpu_mark"].map(_num), "esp_arquitetura": tpu["codename"],
    })
    wiki = pd.read_csv(pasta / "wikipedia_intel_cpulist.csv", encoding="utf-8-sig", dtype=str)
    base = wiki["freq"].map(_num).where(~wiki["freq"].fillna("").str.contains("MHz"), wiki["freq"].map(_num) / 1000)
    turbo = wiki["turbo"].fillna("").map(lambda t: _num(re.findall(r"[\d.]+", t)[-1]) if re.findall(r"[\d.]+", t) else np.nan)
    b = pd.DataFrame({
        "chave": ("Intel " + wiki["model"]).map(chave_cpu), "nome_base": wiki["model"], "fonte": "Wikipedia",
        "esp_lancamento": wiki["date"].map(_ano), "esp_tdp_w": wiki["tdp"].map(_num), "esp_processo_nm": np.nan,
        "esp_clock_ghz": turbo.fillna(base), "esp_unidades": wiki["cores"].map(_num),
        "esp_cache_mb": wiki["l3"].map(_num), "esp_passmark": np.nan, "esp_arquitetura": None,
    })
    t = pd.concat([a, b[~b["chave"].isin(a["chave"])]], ignore_index=True)
    return t[t["chave"] != ""].drop_duplicates("chave").assign(peca="cpu")


def tabela_gpu (pasta: Path = PASTA) -> pd.DataFrame:
    g = pd.DataFrame(json.loads((pasta / "techpowerup_gpu.json").read_text(encoding="utf-8")))
    t = pd.DataFrame({
        "chave": g["name"].map(chave_gpu), "nome_base": g["name"], "fonte": "TechPowerUp",
        "esp_lancamento": g["releaseDate"].map(_ano), "esp_tdp_w": g["tdp"], "esp_processo_nm": g["processSize"],
        "esp_clock_ghz": g["boostClock"].fillna(g["baseClock"]) / 1000, "esp_unidades": g["shaders"],
        "esp_cache_mb": g["l2Cache"], "esp_fp32_tflops": g["fp32"], "esp_banda_gbs": g["memoryBandwidth"],
        "esp_memoria_gb": g["memorySize"], "esp_rt_cores": g["rtCores"].fillna(0.0).where(g["vendor"] != "amd"),
        "esp_arquitetura": g["architecture"],
    })
    sem = t.assign(chave=g["name"].str.replace(" Graphics", "", regex=False).map(chave_gpu))
    t = pd.concat([t.assign(_ordem=0), sem.assign(_ordem=1)], ignore_index=True)
    curta = t.assign(chave=t["nome_base"].str.replace(r"\s+\d+(\.\d+)?\s*GB$", "", regex=True).map(chave_gpu), _ordem=2)
    t = pd.concat([t, curta], ignore_index=True).sort_values(["_ordem", "esp_lancamento"], kind="stable")
    return t[t["chave"] != ""].drop_duplicates("chave").drop(columns="_ordem").assign(peca="gpu")


def montar (pasta: Path = PASTA) -> pd.DataFrame:
    return pd.concat([tabela_cpu(pasta), tabela_gpu(pasta)], ignore_index=True)


def carregar (caminho: Path = SAIDA) -> pd.DataFrame:
    if not caminho.exists():
        t = montar()
        caminho.parent.mkdir(parents=True, exist_ok=True)
        t.to_parquet(caminho, index=False)
    return pd.read_parquet(caminho)


def acrescentar (df: pd.DataFrame, tabela: pd.DataFrame) -> pd.DataFrame:
    cpu = tabela[tabela["peca"] == "cpu"].set_index("chave")
    gpu = tabela[tabela["peca"] == "gpu"].set_index("chave")
    eh_cpu = df["dispositivo_tipo"].astype(str).str.upper().eq("CPU").to_numpy()
    k_cpu = df["dispositivo_nome"].map(chave_cpu)
    k_gpu = df["dispositivo_nome"].map(chave_gpu)
    disp = pd.DataFrame(index=df.index, columns=list(DISPOSITIVO), dtype=object)
    achou_cpu = cpu.reindex(k_cpu)[list(DISPOSITIVO)].set_axis(df.index)
    achou_gpu = gpu.reindex(k_gpu)[list(DISPOSITIVO)].set_axis(df.index)
    disp.loc[eh_cpu] = achou_cpu.loc[eh_cpu].to_numpy()
    disp.loc[~eh_cpu] = achou_gpu.loc[~eh_cpu].to_numpy()
    host = cpu.reindex(df["cpu_nome"].map(chave_cpu))[["esp_passmark", "esp_clock_ghz", "esp_lancamento"]]
    host = host.set_axis(list(HOSPEDEIRA), axis=1).set_axis(df.index)
    out = df.drop(columns=[c for c in COLUNAS if c in df.columns])
    numericas = [c for c in DISPOSITIVO if c != "esp_arquitetura"]
    disp[numericas] = disp[numericas].apply(pd.to_numeric, errors="coerce")
    return pd.concat([out, disp, host.astype(float)], axis=1)


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--medicoes", type=Path, default=RAIZ / "data" / "processed" / "medicoes_publicas.parquet")
    args = ap.parse_args()
    t = montar()
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(SAIDA, index=False)
    print(f"tabela: {(t['peca'] == 'cpu').sum():,} chaves de CPU, {(t['peca'] == 'gpu').sum():,} de GPU -> {SAIDA}")
    if args.medicoes.exists():
        d = acrescentar(pd.read_parquet(args.medicoes), t)
        d["casou"] = d["esp_unidades"].notna()
        print("\nfração de linhas com especificação do dispositivo:")
        print(d.groupby(["fonte", "dispositivo_tipo"])["casou"].mean().round(3).unstack(0).to_string())
        print(f"CPU hospedeira com PassMark: {d['host_passmark'].notna().mean():.1%}")
        sem = d.loc[~d["casou"], "dispositivo_nome"].value_counts().head(15)
        print("\nnomes sem casamento mais frequentes:\n" + sem.to_string())


if __name__ == "__main__":
    main()
