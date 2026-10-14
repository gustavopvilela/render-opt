"""Monta a tabela de envios individuais do pts/blender a partir dos XMLs baixados.
uso: python tabela.py saida.csv pasta_xml [pasta_xml ...]
Uma linha por (envio, sistema, teste do Blender). Nada é descartado: todos os campos do
XML entram, os campos compostos são também desmembrados em colunas."""
import csv, hashlib, json, os, re, statistics, sys
import xml.etree.ElementTree as ET

def txt(el, tag):
    x = el.find(tag)
    return (x.text or "").strip() if x is not None and x.text else ""

def pares(s):
    """'Processor: X, Memory: Y' -> {'Processor': 'X', ...}; vírgulas dentro de valores são mantidas."""
    out, chave = {}, None
    for parte in re.split(r", (?=[A-Z][A-Za-z0-9 /\-]*: )", s or ""):
        m = re.match(r"([A-Z][A-Za-z0-9 /\-]*): (.*)$", parte, re.S)
        if m:
            chave = m.group(1).strip()
            out[chave] = m.group(2).strip()
        elif chave:
            out[chave] += ", " + parte
    return out

def nums(s):
    v = []
    for x in (s or "").split(":"):
        try:
            v.append(float(x))
        except ValueError:
            pass
    return v

def fmt(v):
    return ";".join(f"{x:g}" for x in v)

def processador(p):
    d = {}
    m = re.search(r"\((\d+) Cores?(?: / (\d+) Threads)?\)", p)
    if m:
        d["cpu_nucleos"] = m.group(1)
        d["cpu_threads"] = m.group(2) or ""
    m = re.search(r"@ ([\d.]+)GHz", p)
    if m:
        d["cpu_freq_ghz"] = m.group(1)
    m = re.match(r"(\d+) x ", p)
    d["cpu_soquetes"] = m.group(1) if m else ("1" if p else "")
    d["cpu_modelo"] = re.sub(r"^\d+ x |\s*@.*$|\s*\(.*$", "", p).strip()
    return d

def gpu(g):
    d = {}
    m = re.search(r"(\d+)(MB|GB)\b", g)
    if m:
        d["gpu_memoria_mb"] = str(int(m.group(1)) * (1024 if m.group(2) == "GB" else 1))
    m = re.search(r"\((\d+)/(\d+)MHz\)", g)
    if m:
        d["gpu_clock_nucleo_mhz"], d["gpu_clock_memoria_mhz"] = m.group(1), m.group(2)
    m = re.match(r"(\d+) x ", g)
    d["gpu_quantidade"] = m.group(1) if m else ("1" if g else "")
    return d

def cena_backend(desc, args):
    cena = backend = ""
    m = re.search(r"Blend File: ([^-]+?)(?: - |$)", desc)
    if m:
        cena = m.group(1).strip()
    m = re.search(r"Compute: (.+)$", desc)
    if m:
        backend = m.group(1).strip()
    if not cena:
        m = re.search(r"([\w\-]+)\.blend", args)
        cena = m.group(1) if m else ""
    if not backend:
        m = re.search(r"--cycles-device (\S+)", args)
        backend = m.group(1) if m else ""
    return cena, backend

def main():
    saida, pastas = sys.argv[1], sys.argv[2:]
    linhas, erros = [], []
    for pasta in pastas:
        for nome in sorted(os.listdir(pasta)):
            if not nome.endswith(".xml"):
                continue
            rid = nome[:-4]
            try:
                raiz = ET.parse(os.path.join(pasta, nome)).getroot()
            except ET.ParseError as e:
                erros.append((rid, str(e)))
                continue
            g = raiz.find("Generated")
            ger = {k: txt(g, t) for k, t in [
                ("envio_titulo", "Title"), ("envio_ultima_modificacao", "LastModified"),
                ("envio_cliente_pts", "TestClient"), ("envio_descricao", "Description"),
                ("envio_notas", "Notes"), ("envio_tags_internas", "InternalTags"),
                ("envio_id_referencia", "ReferenceID"),
                ("envio_variaveis_ambiente", "PreSetEnvironmentVariables")]} if g is not None else {}
            sistemas = {}
            for s in raiz.findall("System"):
                ident = txt(s, "Identifier")
                hw, sw = txt(s, "Hardware"), txt(s, "Software")
                d = {"sistema_identificador": ident, "hardware_texto": hw, "software_texto": sw,
                     "usuario": txt(s, "User"), "data_teste": txt(s, "TimeStamp"),
                     "sistema_versao_cliente": txt(s, "TestClientVersion"),
                     "sistema_notas": txt(s, "Notes"), "sistema_json": txt(s, "JSON")}
                for k, v in pares(hw).items():
                    d["hw_" + k] = v
                for k, v in pares(sw).items():
                    d["sw_" + k] = v
                try:
                    for k, v in (json.loads(d["sistema_json"]) if d["sistema_json"] else {}).items():
                        d["sys_" + k] = v if isinstance(v, str) else json.dumps(v)
                except ValueError:
                    pass
                d.update(processador(d.get("hw_Processor", "")))
                d.update(gpu(d.get("hw_Graphics", "")))
                sistemas[ident] = d
            for r in raiz.findall("Result"):
                perfil = txt(r, "Identifier")
                if not perfil.startswith("pts/blender"):
                    continue
                desc, args = txt(r, "Description"), txt(r, "Arguments")
                cena, backend = cena_backend(desc, args)
                base = {"id_resultado": rid, "perfil_teste": perfil,
                        "versao_perfil": perfil.split("-")[-1] if "-" in perfil else "",
                        "teste_titulo": txt(r, "Title"), "versao_blender": txt(r, "AppVersion"),
                        "teste_descricao": desc, "argumentos": args, "cena": cena, "backend": backend,
                        "backend_normalizado": {"NVIDIA CUDA": "CUDA", "NVIDIA OptiX": "OptiX",
                                                "Radeon HIP": "HIP", "Radeon HIP-RT": "HIP-RT",
                                                "Intel oneAPI": "oneAPI", "macOS Metal": "Metal",
                                                "CPU": "CPU-Only"}.get(backend, backend),
                        "escala": txt(r, "Scale"), "proporcao": txt(r, "Proportion"),
                        "formato_exibicao": txt(r, "DisplayFormat"),
                        "teste_anotacao": txt(r, "Annotation")}
                for e in r.findall("Data/Entry"):
                    ident = txt(e, "Identifier")
                    linha = dict(base, **ger)
                    linha.update(sistemas.get(ident, {"sistema_identificador": ident,
                                                      "sistema_sem_descricao": "1"}))
                    raw = nums(txt(e, "RawString"))
                    ej = txt(e, "JSON")
                    linha.update({"valor": txt(e, "Value"), "rodadas_texto": txt(e, "RawString"),
                                  "rodadas": fmt(raw), "n_rodadas": str(len(raw)),
                                  "entrada_json": ej})
                    if raw:
                        linha["rodadas_min"], linha["rodadas_max"] = f"{min(raw):g}", f"{max(raw):g}"
                        if len(raw) > 1:
                            linha["rodadas_desvio_padrao"] = f"{statistics.stdev(raw):.4g}"
                            linha["rodadas_cv_pct"] = f"{100*statistics.stdev(raw)/statistics.mean(raw):.3g}"
                    try:
                        jd = json.loads(ej) if ej else {}
                    except ValueError:
                        jd = {}
                    for k, v in jd.items():
                        linha["entrada_" + k] = v if isinstance(v, str) else json.dumps(v)
                    tt = nums(jd.get("test-run-times", "")) if isinstance(jd.get("test-run-times"), str) else []
                    if tt:
                        linha["tempos_totais"] = fmt(tt)
                        if len(tt) == len(raw):
                            dif = [a - b for a, b in zip(tt, raw)]
                            linha["carregamento_por_rodada"] = fmt([round(x, 3) for x in dif])
                            linha["carregamento_medio_s"] = f"{statistics.mean(dif):.3f}"
                    chave = "|".join([linha.get("hardware_texto", ""), linha.get("software_texto", ""),
                                      linha.get("data_teste", ""), perfil, args, linha["rodadas_texto"]])
                    linha["chave_duplicata"] = hashlib.sha1(chave.encode()).hexdigest()[:16]
                    linhas.append(linha)
    vistos = {}
    for l in linhas:
        vistos.setdefault(l["chave_duplicata"], l["id_resultado"])
        l["duplicata_de"] = "" if vistos[l["chave_duplicata"]] == l["id_resultado"] else vistos[l["chave_duplicata"]]
    ordem = ["id_resultado", "sistema_identificador", "data_teste", "perfil_teste", "versao_perfil",
             "versao_blender", "cena", "backend", "backend_normalizado", "valor", "escala", "n_rodadas", "rodadas",
             "rodadas_min", "rodadas_max", "rodadas_desvio_padrao", "rodadas_cv_pct", "tempos_totais",
             "carregamento_por_rodada", "carregamento_medio_s", "cpu_modelo", "cpu_soquetes",
             "cpu_nucleos", "cpu_threads", "cpu_freq_ghz", "gpu_quantidade", "gpu_memoria_mb",
             "gpu_clock_nucleo_mhz", "gpu_clock_memoria_mhz", "chave_duplicata", "duplicata_de"]
    resto = sorted({k for l in linhas for k in l} - set(ordem))
    cols = ordem + [c for c in resto if c.startswith(("hw_", "sw_", "sys_"))] + \
        [c for c in resto if not c.startswith(("hw_", "sw_", "sys_"))]
    with open(saida, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(linhas)
    print(len(linhas), "linhas,", len(cols), "colunas,", len({l["id_resultado"] for l in linhas}),
          "envios,", sum(1 for l in linhas if l["duplicata_de"]), "duplicatas,", len(erros), "erros XML")

if __name__ == "__main__":
    main()
