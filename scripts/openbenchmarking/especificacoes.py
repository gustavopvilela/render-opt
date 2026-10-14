"""Liga cada processador e placa de vídeo da tabela completada às especificações do TechPowerUp.
uso: python especificacoes.py tabela_completa.csv cpu.csv wiki_intel.csv all-gpus.json saida.csv relatorio.csv

Fontes (cópias em data/raw/especificacoes/):
  cpu.csv       github.com/lijieyu233/cpu-gpu-data-crawler (out/cpu.csv), lista do TechPowerUp CPU Database + PassMark
  wiki_intel.csv tabelas {{cpulist}} da Wikipédia em inglês (gerado por wikipedia_cpulist.py), 2ª fonte p/ Intel
  all-gpus.json github.com/RightNow-AI/RightNow-GPU-Database (data/all-gpus.json), TechPowerUp GPU Database, Apache-2.0
Colunas novas: cpu_spec_* e gpu_spec_*, mais cpu_spec_casamento / gpu_spec_casamento:
  exato      nome normalizado igual (CPU: TechPowerUp)
  exato_wikipedia  CPU não está no TechPowerUp, achada na Wikipédia
  exato_primeiro   GPU com várias variantes e sem memória no envio para desempatar; usada a variante-base
  memoria    vários modelos com o mesmo nome; escolhido o de memória igual à do envio
  ambiguo    o PTS juntou vários modelos num nome só (ex.: 'RX 470/480/570'); não preenchido
  sem_match  modelo não encontrado na base"""
import csv, json, re, sys
from collections import defaultdict

csv.field_size_limit(10**9)


def chave_cpu(m):
    m = re.sub(r"\((R|TM)\)|®|™", "", m or "")
    m = re.sub(r"\b(AMD|Intel|Processor|CPU|with Radeon Graphics|w/ Radeon \S+ Graphics)\b", "", m, flags=re.I)
    m = re.sub(r"\b\d+-Cores?\b|\b(Dual|Triple|Quad|Six|Eight|Ten|Twelve|Sixteen)-Core\b", "", m, flags=re.I)
    m = re.sub(r"\s0\s*$", "", m)  # o PTS escreve 'Xeon E5-2667 0' para a 1ª geração (sem 'v')
    return re.sub(r"[^a-z0-9]", "", m.lower())


PARCEIRAS = r"(eVGA|EVGA|ASUS|MSI|Gigabyte|Zotac|XFX|Sapphire|PowerColor|ASRock|Palit|PNY|Gainward|Inno3D|Colorful|Galax|KFA2|Dell|HP|Lenovo|Acer|Biostar|Yeston|Sparkle|Elsa)"


def chave_gpu(m):
    m = re.sub(r"\((R|TM)\)|®|™", "", m or "")
    # híbrida 'Intel Iris Xe + NVIDIA GeForce RTX 3070 Ti Laptop GPU': fica a dedicada
    partes = m.split(" + ")
    m = next((x for x in partes if re.search(r"NVIDIA|GeForce|Radeon RX|Radeon Pro|\bArc", x)), partes[-1])
    m = m.replace("Arctm", "Arc")
    m = re.sub(r"\s+(DG2|DG1)$", "", m.strip())
    m = re.sub(r"^" + PARCEIRAS + r"\s+", "", m.strip(), flags=re.I)
    m = m.replace("Laptop GPU", "Mobile").replace("with Max-Q Design", "Max-Q")
    m = re.sub(r"^(NVIDIA|AMD|ATI|Intel|Apple)\s+", "", m.strip())
    m = re.sub(r"\bGraphics\b", "", m)
    m = re.sub(r"\[.*?\]|\(.*?\)", "", m)
    return re.sub(r"[^a-z0-9]", "", m.lower())


def num_ghz(s):
    """Primeiro número do texto: '2.0 GHz' -> '2.0', '27.50' -> '27.50'."""
    m = re.search(r"[\d.]+", s or "")
    return m.group(0) if m else ""


CPU_CAMPOS = ["fonte", "model", "codename", "cores", "threads", "base_clock_ghz", "boost_clock_ghz", "socket",
              "process_nm", "l3_cache_mb", "tdp_w", "released_date", "passmark_cpu_mark",
              "passmark_single_thread", "price_usd", "url"]
GPU_CAMPOS = ["name", "manufacturer", "gpuName", "architecture", "generation", "foundry", "processSize",
              "transistors", "dieSize", "releaseDate", "busInterface", "baseClock", "boostClock",
              "memoryClock", "memorySize", "memoryType", "memoryBus", "memoryBandwidth", "shaders", "tmus",
              "rops", "sms", "computeUnits", "executionUnits", "tensorCores", "rtCores", "l1Cache", "l2Cache",
              "tdp", "pixelRate", "textureRate", "fp16", "fp32", "fp64", "cuda", "openCL", "vulkan", "url"]


def main():
    entrada, cpu_csv, wiki_csv, gpu_json, saida, relatorio = sys.argv[1:7]
    cpus = defaultdict(list)
    for c in csv.DictReader(open(cpu_csv, encoding="utf-8-sig")):
        cpus[chave_cpu(c["brand"] + " " + c["model"])].append(c)
    wiki = {}
    for c in csv.DictReader(open(wiki_csv, encoding="utf-8-sig")):
        k = chave_cpu("Intel " + c["model"])
        if k in wiki:
            continue
        base = ""
        if c.get("freq"):
            base = num_ghz(c["freq"])
            if base and "MHz" in c["freq"]:
                base = f"{float(base) / 1000:g}"
        elif c.get("mult"):
            try:
                base = f"{float(c['mult']) / 10:g}"
            except ValueError:
                pass
        turbo = re.findall(r"[\d.]+", c.get("turbo", ""))
        wiki[k] = {"fonte": "Wikipedia", "model": c["model"], "cores": c.get("cores", "").strip(),
                   "threads": c.get("threads", "").strip(), "base_clock_ghz": base,
                   "boost_clock_ghz": turbo[-1] if turbo else "", "socket": c.get("sock", ""),
                   "l3_cache_mb": num_ghz(c.get("l3", "")), "tdp_w": num_ghz(c.get("tdp", "")),
                   "released_date": c.get("date", ""), "price_usd": c.get("price", ""),
                   "url": "https://en.wikipedia.org/wiki/" + c["pagina_wikipedia"].replace(" ", "_")}
    gpus = defaultdict(list)
    for g in json.load(open(gpu_json, encoding="utf-8")):
        gpus[chave_gpu(g["name"])].append(g)
        # TechPowerUp chama integradas de 'UHD Graphics 630'; o PTS escreve 'UHD 630'
        gpus[chave_gpu(g["name"].replace(" Graphics", ""))].append(g)

    cache_cpu, cache_gpu, rel = {}, {}, {}

    def achar_cpu(modelo):
        if modelo in cache_cpu:
            return cache_cpu[modelo]
        k = chave_cpu(modelo)
        lst = cpus.get(k, [])
        if lst:
            res = (dict(lst[0], fonte="TechPowerUp", url=lst[0]["techpowerup_url"]), "exato")
        elif k in wiki:
            res = (wiki[k], "exato_wikipedia")
        else:
            res = (None, "sem_match")
        cache_cpu[modelo] = res
        return res

    def achar_gpu(modelo, mem_mb):
        ck = (modelo, mem_mb)
        if ck in cache_gpu:
            return cache_gpu[ck]
        if "/" in modelo:
            res = (None, "ambiguo")
        else:
            k = chave_gpu(modelo)
            lst = list({id(g): g for g in gpus.get(k, [])}.values())
            if not lst and k:
                # 'GeForce GTX 1060' no envio x 'GeForce GTX 1060 6 GB' na base: variantes de memória
                lst = list({id(g): g for kk, gs in gpus.items() if kk.startswith(k)
                            and re.fullmatch(r"\d+gb", kk[len(k):]) for g in gs}.values())
            if not lst:
                res = (None, "sem_match")
            elif len(lst) == 1:
                res = (lst[0], "exato")
            else:
                mem = float(mem_mb) / 1024 if mem_mb else None
                iguais = [g for g in lst if mem and g.get("memorySize") and abs(g["memorySize"] - mem) < 0.6]
                # sem memória para desempatar: o modelo-base (nome mais curto, lançamento mais antigo)
                res = (iguais[0], "memoria") if iguais else \
                      (sorted(lst, key=lambda g: (len(g["name"]), g.get("releaseDate") or ""))[0], "exato_primeiro")
        cache_gpu[ck] = res
        return res

    leitor = csv.DictReader(open(entrada, encoding="utf-8-sig"))
    cols = leitor.fieldnames + ["cpu_spec_casamento"] + ["cpu_spec_" + c for c in CPU_CAMPOS] + \
        ["gpu_spec_casamento"] + ["gpu_spec_" + c for c in GPU_CAMPOS]
    with open(saida, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for x in leitor:
            c, how = achar_cpu(x["cpu_modelo"])
            x["cpu_spec_casamento"] = how
            for k in CPU_CAMPOS:
                x["cpu_spec_" + k] = c.get(k, "") if c else ""
            rel.setdefault(("cpu", x["cpu_modelo"], "", how, c["model"] if c else ""), 0)
            rel[("cpu", x["cpu_modelo"], "", how, c["model"] if c else "")] += 1
            if x["gpu_tipo"] in ("dedicada", "integrada"):
                g, how = achar_gpu(x["gpu_modelo"], x["gpu_memoria_mb_completo"])
            else:
                g, how = None, "sem_gpu"
            x["gpu_spec_casamento"] = how
            for k in GPU_CAMPOS:
                v = g.get(k, "") if g else ""
                x["gpu_spec_" + k] = "" if v is None else v
            if how != "sem_gpu":
                chave = ("gpu", x["gpu_modelo"], x["gpu_memoria_mb_completo"], how, g["name"] if g else "")
                rel[chave] = rel.get(chave, 0) + 1
            w.writerow(x)
    with open(relatorio, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["peca", "modelo_no_envio", "memoria_mb_envio", "casamento", "modelo_na_base", "linhas"])
        for k, v in sorted(rel.items(), key=lambda t: -t[1]):
            w.writerow(list(k) + [v])
    for tipo in ("cpu", "gpu"):
        tot = defaultdict(int)
        mods = defaultdict(int)
        for k, v in rel.items():
            if k[0] == tipo:
                tot[k[3]] += v
                mods[k[3]] += 1
        print(tipo, "linhas:", dict(tot), "modelos:", dict(mods))


if __name__ == "__main__":
    main()
