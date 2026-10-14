"""Gera uma cópia da tabela de envios com dados completados.
uso: python completar.py entrada.csv saida.csv

As colunas originais não são alteradas. Cada dado completado vai para uma coluna nova
<campo>_completo, acompanhada de <campo>_origem, que diz de onde veio o valor:
  envio          -> estava no próprio envio
  mesmo_modelo   -> copiado de outro envio com o mesmo modelo de peça (valor mais frequente)
  regra          -> deduzido do formato do texto do PTS (explicado em cada função)
  vazio          -> não foi possível completar
Também cria colunas derivadas (fabricante, tipo de GPU, memória em GB, id de máquina etc.)."""
import csv, collections, hashlib, re, sys

csv.field_size_limit(10**9)

PARCEIRAS = ("ASUS", "MSI", "Gigabyte", "GIGABYTE", "EVGA", "Zotac", "ZOTAC", "XFX", "Sapphire",
             "PowerColor", "ASRock", "Palit", "PNY", "Gainward", "Inno3D", "Colorful", "Galaxy",
             "KFA2", "Dell", "HP", "Lenovo", "Acer", "Biostar", "Yeston", "Sparkle", "Matrox",
             "AMD/ATI", "Elsa")
SEM_GPU = re.compile(r"^(llvmpipe|LLVMpipe|ASPEED|astdrmfb|mgag200drmfb|amdgpudrmfb|efifb|"
                     r"simpledrmfb|VMware SVGA|Microsoft Basic|Matrox|Cirrus|virtio|EFI VGA|"
                     r"inteldrmfb|nouveaufb|bochs|QXL|Red Hat|SVGA3D|vesa|simple-framebuffer)", re.I)


def gpu_modelo(g):
    """Nome da GPU sem fabricante da placa, memória e clocks: 'ASUS NVIDIA GeForce RTX 3060 Ti 8GB' -> 'NVIDIA GeForce RTX 3060 Ti'."""
    g = re.sub(r"^\d+ x ", "", g or "").strip()
    g = re.sub(r"\s*\(\d+/\d+MHz\)", "", g)
    g = re.sub(r"\s+\d+(MB|GB)\b.*$", "", g)
    for p in PARCEIRAS:
        if g.startswith(p + " "):
            g = g[len(p) + 1:]
            break
    return g.strip()


def gpu_fabricante(m):
    for chave, nome in [("NVIDIA", "NVIDIA"), ("GeForce", "NVIDIA"), ("Quadro", "NVIDIA"), ("Tesla", "NVIDIA"),
                        ("Radeon", "AMD"), ("AMD", "AMD"), ("FirePro", "AMD"), ("Instinct", "AMD"),
                        ("Intel", "Intel"), ("Arc", "Intel"), ("Apple", "Apple")]:
        if chave in m:
            return nome
    return ""


def cpu_fabricante(m):
    for chave, nome in [("AMD", "AMD"), ("Intel", "Intel"), ("Apple", "Apple"), ("ARM", "ARM"),
                        ("Ampere", "Ampere"), ("Cortex", "ARM"), ("Neoverse", "ARM"), ("POWER", "IBM"),
                        ("Qualcomm", "Qualcomm"), ("Snapdragon", "Qualcomm"), ("NVIDIA", "NVIDIA"),
                        ("Cavium", "Cavium"), ("ThunderX", "Cavium"), ("Loongson", "Loongson")]:
        if chave in m:
            return nome
    return ""


def memoria(s):
    """'2 x 16GB DDR5-6000MT/s ...' -> total_gb=32, modulos=2, tipo=DDR5, mts=6000; '16384MB' -> 16."""
    d = {"memoria_total_gb": "", "memoria_modulos": "", "memoria_tipo": "", "memoria_mts": ""}
    if not s:
        return d
    total = 0.0
    for parte in s.split(" + "):
        m = re.match(r"\s*(?:(\d+) x )?([\d.]+)\s*(MB|GB|TB)", parte)
        if m:
            n = int(m.group(1) or 1)
            v = float(m.group(2)) * {"MB": 1 / 1024, "GB": 1, "TB": 1024}[m.group(3)]
            total += n * v
            if m.group(1):
                d["memoria_modulos"] = str(int(d["memoria_modulos"] or 0) + n)
    if total:
        d["memoria_total_gb"] = f"{total:g}" if total >= 1 else f"{total:.3g}"
    m = re.search(r"\b(LP)?(DDR\d[A-Z]?|HBM\d?e?)\b", s)
    if m:
        d["memoria_tipo"] = m.group(0)
    m = re.search(r"-(\d{3,5})\s*(MT/s|MHz)", s)
    if m:
        d["memoria_mts"] = m.group(1)
    return d


def so_familia(os_):
    o = (os_ or "").lower()
    for chave, nome in [("windows", "Windows"), ("macos", "macOS"), ("mac os", "macOS"), ("os x", "macOS"),
                        ("freebsd", "BSD"), ("openbsd", "BSD"), ("netbsd", "BSD"), ("dragonfly", "BSD"),
                        ("solaris", "Solaris"), ("illumos", "Solaris"), ("omnios", "Solaris")]:
        if chave in o:
            return nome
    return "Linux" if o else ""


def moda(c):
    return c.most_common(1)[0][0] if c else ""


def main():
    entrada, saida = sys.argv[1], sys.argv[2]
    r = list(csv.DictReader(open(entrada, encoding="utf-8-sig")))
    cols_orig = list(r[0].keys())

    # 1) derivados por linha
    for x in r:
        proc = x["hw_Processor"]
        x["cpu_fabricante"] = cpu_fabricante(x["cpu_modelo"])
        # O PTS mostra "(N Cores / M Threads)" quando threads != núcleos; só "(N Cores)" quando são
        # iguais ou em versões antigas do cliente, que contavam threads como núcleos. Nos dois casos
        # N é o número de threads lógicas.
        x["_threads_explicito"] = "Thread" in proc
        mg = gpu_modelo(x["hw_Graphics"])
        x["gpu_modelo"] = mg
        sem = bool(SEM_GPU.match(mg)) or not mg
        x["gpu_tipo"] = ("sem GPU / vídeo básico" if sem else
                         "integrada" if re.search(r"(Intel (HD|UHD|Iris|Graphics|Xe)\b|Radeon (Graphics|Vega \d+ Graphics|\d{3}M)\b|"
                                                  r"Renoir|Cezanne|Rembrandt|Phoenix|Raphael|Barcelo|Lucienne|Picasso|Raven|"
                                                  r"Strix|Hawk Point|Granite Ridge|Lunar|Meteor|Arrow Lake|Apple M\d)", mg)
                         else "dedicada")
        x["gpu_fabricante"] = "" if sem else gpu_fabricante(mg)
        x.update(memoria(x["hw_Memory"]))
        x["so_familia"] = so_familia(x["sw_OS"])
        if x["backend_normalizado"] in ("CPU-Only", ""):
            x["dispositivo_render"] = x["cpu_modelo"]
        elif sem:
            # Render em GPU, mas o PTS só viu o adaptador de vídeo básico (máquina virtual, servidor com
            # GPU de computação): a GPU que renderizou não está identificada no envio.
            x["dispositivo_render"] = ""
            x["gpu_tipo"] = "não identificada (render em GPU)"
        else:
            x["dispositivo_render"] = mg
        x["ano_teste"] = x["data_teste"][:4]
        m = re.match(r"(\d+)\.(\d+)", x["versao_blender"])
        x["versao_blender_num"] = f"{int(m.group(1))}.{int(m.group(2)):02d}" if m else ""
        x["maquina_id"] = hashlib.sha1(x["hardware_texto"].encode()).hexdigest()[:12]
        x["maquina_software_id"] = hashlib.sha1((x["hardware_texto"] + "|" + x["software_texto"]).encode()).hexdigest()[:12]

    # 2) tabelas por modelo, só com valores vindos do próprio envio
    por_cpu = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    por_gpu = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for x in r:
        k = x["cpu_modelo"]
        if x["cpu_freq_ghz"]:
            por_cpu[k]["freq"][x["cpu_freq_ghz"]] += 1
        if x["_threads_explicito"]:
            por_cpu[k]["nucleos"][x["cpu_nucleos"]] += 1
            por_cpu[k]["threads"][x["cpu_threads"]] += 1
        if x["gpu_tipo"] != "sem GPU / vídeo básico":
            g = x["gpu_modelo"]
            for campo, col in [("mem", "gpu_memoria_mb"), ("clk", "gpu_clock_nucleo_mhz"),
                               ("clkm", "gpu_clock_memoria_mhz"), ("cores", "sys_graphics-compute-cores"),
                               ("bar1", "sys_bar1-visible-vram")]:
                if x.get(col):
                    por_gpu[g][campo][x[col]] += 1

    def completar(x, nome, valor_envio, tabela, chave, campo, regra=None):
        if valor_envio:
            x[nome + "_completo"], x[nome + "_origem"] = valor_envio, "envio"
        elif regra:
            x[nome + "_completo"], x[nome + "_origem"] = regra, "regra"
        elif tabela.get(chave) and tabela[chave][campo]:
            x[nome + "_completo"], x[nome + "_origem"] = moda(tabela[chave][campo]), "mesmo_modelo"
        else:
            x[nome + "_completo"], x[nome + "_origem"] = "", "vazio"

    for x in r:
        c = x["cpu_modelo"]
        # threads: sem "/ Threads" o número mostrado já é a contagem lógica (regra).
        completar(x, "cpu_threads", x["cpu_threads"] if x["_threads_explicito"] else "", por_cpu, c, "threads",
                  regra=None if x["_threads_explicito"] else x["cpu_nucleos"])
        # núcleos físicos: só confiável quando o envio separa núcleos e threads; senão, outro envio do mesmo modelo.
        completar(x, "cpu_nucleos_fisicos", x["cpu_nucleos"] if x["_threads_explicito"] else "", por_cpu, c, "nucleos")
        completar(x, "cpu_freq_ghz", x["cpu_freq_ghz"], por_cpu, c, "freq")
        g = x["gpu_modelo"]
        if x["gpu_tipo"] == "sem GPU / vídeo básico":
            # "llvmpipe 188GB" é a RAM do sistema, não memória de vídeo.
            for nome in ("gpu_memoria_mb", "gpu_clock_nucleo_mhz", "gpu_clock_memoria_mhz", "gpu_nucleos_computacao", "gpu_bar1_vram"):
                x[nome + "_completo"], x[nome + "_origem"] = "", "sem_gpu"
            continue
        completar(x, "gpu_memoria_mb", x["gpu_memoria_mb"], por_gpu, g, "mem")
        completar(x, "gpu_clock_nucleo_mhz", x["gpu_clock_nucleo_mhz"], por_gpu, g, "clk")
        completar(x, "gpu_clock_memoria_mhz", x["gpu_clock_memoria_mhz"], por_gpu, g, "clkm")
        completar(x, "gpu_nucleos_computacao", x.get("sys_graphics-compute-cores", ""), por_gpu, g, "cores")
        completar(x, "gpu_bar1_vram", x.get("sys_bar1-visible-vram", ""), por_gpu, g, "bar1")

    # 3) por máquina: mesma máquina (mesmo texto de hardware) em outros envios pode ter o campo preenchido
    por_maquina = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    campos_maquina = ["sw_Display Driver", "sw_OpenCL", "sw_Vulkan", "sw_OpenGL", "sys_cpu-microcode",
                      "sys_cpu-scaling-governor", "sys_vbios-version", "hw_Monitor", "hw_Audio", "hw_Network"]
    for x in r:
        for col in campos_maquina:
            if x.get(col):
                por_maquina[x["maquina_software_id"] if col.startswith("sw_") else x["maquina_id"]][col][x[col]] += 1
    for x in r:
        for col in campos_maquina:
            nome = col.replace(" ", "_").replace("-", "_")
            chave = x["maquina_software_id"] if col.startswith("sw_") else x["maquina_id"]
            if x.get(col):
                x[nome + "_completo"], x[nome + "_origem"] = x[col], "envio"
            elif por_maquina[chave][col]:
                x[nome + "_completo"], x[nome + "_origem"] = moda(por_maquina[chave][col]), "mesma_maquina"
            else:
                x[nome + "_completo"], x[nome + "_origem"] = "", "vazio"

    novas = ["cpu_fabricante", "gpu_modelo", "gpu_fabricante", "gpu_tipo", "dispositivo_render",
             "memoria_total_gb", "memoria_modulos", "memoria_tipo", "memoria_mts", "so_familia", "ano_teste",
             "versao_blender_num", "maquina_id", "maquina_software_id"]
    completadas = sorted({k for x in r for k in x if k.endswith(("_completo", "_origem"))},
                         key=lambda k: (k.rsplit("_", 1)[0], k.endswith("_origem")))
    cols = cols_orig + novas + completadas
    with open(saida, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(r)
    print(len(r), "linhas,", len(cols), "colunas")
    for k in completadas:
        if k.endswith("_origem"):
            c = collections.Counter(x.get(k, "") for x in r)
            print(f"{k[:-7]:28s}", dict(c))


if __name__ == "__main__":
    main()
