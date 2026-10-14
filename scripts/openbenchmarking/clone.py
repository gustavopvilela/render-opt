"""Baixa resultados do OpenBenchmarking via API do cliente PTS e guarda só os que têm pts/blender.
uso: python clone.py ids.txt saida_dir [n_threads]"""
import json, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://openbenchmarking.org/f/client.php"
UA = "PhoronixTestSuite/Nesseby"

def pedir(i):
    dados = urllib.parse.urlencode({"r": "clone_openbenchmarking_result", "i": i,
                                    "client_version": "10860"}).encode()
    req = urllib.request.Request(URL, data=dados, headers={"User-Agent": UA})
    for t in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            erro = str(e)
            time.sleep(3 * (t + 1))
    return "ERRO " + erro

def main():
    ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    saida = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(os.path.join(saida, "blender"), exist_ok=True)
    log_path = os.path.join(saida, "log.tsv")
    feitos = set()
    if os.path.exists(log_path):
        feitos = {l.split("\t")[0] for l in open(log_path, encoding="utf-8")}
    pendentes = [i for i in ids if i not in feitos]
    log = open(log_path, "a", encoding="utf-8")

    def um(i):
        txt = pedir(i)
        if txt.startswith("ERRO"):
            return i, "erro", txt[:80]
        try:
            xml = json.loads(txt)["openbenchmarking"]["result"]["composite_xml"]
        except Exception:
            return i, "vazio", txt[:80].replace("\n", " ")
        if "pts/blender" in xml:
            with open(os.path.join(saida, "blender", i + ".xml"), "w", encoding="utf-8") as f:
                f.write(xml)
            return i, "blender", str(xml.count("pts/blender"))
        return i, "outro", ""

    with ThreadPoolExecutor(n) as ex:
        for k, (i, st, info) in enumerate(ex.map(um, pendentes)):
            log.write(f"{i}\t{st}\t{info}\n")
            if k % 50 == 0:
                log.flush()
                print(k, len(pendentes), flush=True)
    log.close()

if __name__ == "__main__":
    main()
