# scripts/analisar_opendata_v4.py
import zipfile, io, json, collections, statistics as st
from pathlib import Path

CACHE = Path("data/raw/opendata-latest.zip")
CENAS = ("monster", "junkshop", "classroom")
BASE = "monster"

vazao = collections.defaultdict(list)
z = zipfile.ZipFile(CACHE)
nome = next(n for n in z.namelist() if n.endswith(".jsonl"))
with z.open(nome) as fh:
    for linha in io.TextIOWrapper(fh, "utf-8"):
        rec = json.loads(linha)
        if rec.get("schema_version") != "v4":
            continue
        for e in rec["data"]:
            spm = e["stats"].get("samples_per_minute")
            if not spm:
                continue
            di = e["device_info"]
            comp = di.get("compute_devices") or []
            dev = (comp[0]["name"] if comp else "?", di["device_type"])
            vazao[(dev, e["scene"]["label"])].append(spm)

# mediana por (dispositivo, cena), exigindo >= 30 medições
med = {k: st.median(v) for k, v in vazao.items() if len(v) >= 30}
por_dev = collections.defaultdict(dict)
for (dev, cena), m in med.items():
    por_dev[dev][cena] = m
completos = {d: v for d, v in por_dev.items() if len(v) == len(CENAS)}
print(f"{len(completos)} dispositivos com cobertura completa")

# razão = vazão(base) / vazão(cena)  ->  maior = cena mais cara
por_tipo = collections.defaultdict(lambda: collections.defaultdict(list))
for (_, tipo), v in completos.items():
    for c in CENAS:
        if c != BASE:
            por_tipo[tipo][c].append(v[BASE] / v[c])

for tipo in sorted(por_tipo, key=lambda t: -len(por_tipo[t][CENAS[1]])):
    linha = por_tipo[tipo]
    n = len(linha[CENAS[1]])
    if n < 15:
        continue
    partes = " | ".join(f"{c}: {st.median(linha[c]):.2f}" for c in CENAS if c != BASE)
    print(f"{tipo:8s} n={n:4d}  {partes}")
