"""Baixa da Wikipédia (inglês) todas as tabelas {{cpulist}} de processadores Intel e grava num CSV.
uso: python wikipedia_cpulist.py saida.csv
Usado como 2ª fonte de especificações, para os processadores que faltam na base do TechPowerUp."""
import csv, json, re, sys, time, urllib.error, urllib.parse, urllib.request

API = "https://en.wikipedia.org/w/api.php"
UA = "render-opt-tcc/1.0 (pesquisa academica)"


def api(**p):
    p.update(format="json", formatversion="2")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(p), headers={"User-Agent": UA})
    for t in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or t == 5:
                raise
            time.sleep(15 * (t + 1))


def limpar(v):
    v = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", v, flags=re.S)
    v = re.sub(r"<br\s*/?>", " / ", v)
    v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", v)
    v = re.sub(r"\{\{(?:[^|}]*\|)?([^}]*)\}\}", r"\1", v)
    v = re.sub(r"<[^>]+>|'''?", "", v)
    return re.sub(r"\s+", " ", v).strip()


def templates(texto):
    """Extrai {{cpulist|...}} respeitando chaves aninhadas."""
    i = 0
    while True:
        i = texto.find("{{cpulist", i)
        if i < 0:
            return
        prof, j = 0, i
        while j < len(texto):
            if texto.startswith("{{", j):
                prof += 1
                j += 2
            elif texto.startswith("}}", j):
                prof -= 1
                j += 2
                if prof == 0:
                    break
            else:
                j += 1
        yield texto[i + 2:j - 2]
        i = j


def campos(t):
    partes, prof, atual = [], 0, ""
    for ch in t:
        if ch in "{[":
            prof += 1
        elif ch in "}]":
            prof -= 1
        if ch == "|" and prof == 0:
            partes.append(atual)
            atual = ""
        else:
            atual += ch
    partes.append(atual)
    d = {}
    for p in partes[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            d[k.strip()] = limpar(v)
    return d


def main():
    paginas = [x["title"] for x in api(action="query", list="embeddedin", eititle="Template:Cpulist",
                                       eilimit="500", einamespace="0")["query"]["embeddedin"]]
    linhas, chaves = [], set()
    for pg in paginas:
        txt = api(action="parse", page=pg, prop="wikitext")["parse"]["wikitext"]
        n = 0
        for t in templates(txt):
            d = campos(t)
            if d.get("model"):
                d["pagina_wikipedia"] = pg
                linhas.append(d)
                chaves |= set(d)
                n += 1
        print(n, pg, flush=True)
        time.sleep(4)
    ordem = ["model", "cores", "threads", "mult", "freq", "turbo", "l2", "l3", "tdp", "sock", "date",
             "price", "pagina_wikipedia"]
    cols = [c for c in ordem if c in chaves] + sorted(chaves - set(ordem))
    with open(sys.argv[1], "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(linhas)
    print(len(linhas), "processadores")


if __name__ == "__main__":
    main()
