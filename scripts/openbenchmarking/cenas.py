"""Lê, com o Blender, o trabalho (amostras x pixels) de cada cena que o pts/blender renderiza.
uso: python cenas.py <blender> <saida.csv> <arquivo.blend> [<arquivo.blend> ...]

Abra cada .blend DO JEITO QUE O PTS O DEIXA: o install.sh move só o .blend principal para a
pasta do usuário e apaga o resto (rm -rf benchmark). No classroom de 2016 isso deixa as 11
bibliotecas ligadas (assets/*.blend) para trás. Copie o arquivo para uma pasta vazia antes de
rodar, para que a contagem de bibliotecas ausentes saia igual à do benchmark."""
import csv, hashlib, json, subprocess, sys, tempfile
from pathlib import Path

CENA = {"bmw27": "BMW27", "classroom": "Classroom", "fishy_cat": "Fishy Cat",
        "pavillon_barcelone": "Pabellon Barcelona", "barbershop_interior": "Barbershop", "junkshop": "Junkshop"}

LER = r"""
import bpy, json
s = bpy.context.scene; c = s.cycles; r = s.render
libs = list(bpy.data.libraries)
print("JSON" + json.dumps({"amostras": c.samples, "res_x": r.resolution_x, "res_y": r.resolution_y,
      "res_pct": r.resolution_percentage, "adaptativa": c.use_adaptive_sampling, "denoise": c.use_denoising,
      "bibliotecas": len(libs), "bibliotecas_ausentes": sum(bool(l.is_missing) for l in libs)}))
"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main():
    blender, saida, arquivos = sys.argv[1], sys.argv[2], sys.argv[3:]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(LER)
    linhas = []
    for a in map(Path, arquivos):
        base = a.stem.removesuffix("_cpu").removesuffix("_gpu")
        out = subprocess.run([blender, "-b", str(a), "--factory-startup", "--disable-autoexec", "--python", f.name],
                             capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        dados = json.loads(next(l[4:] for l in out.splitlines() if l.startswith("JSON")))
        pct = dados["res_pct"] / 100
        linhas.append({"cena_pts": CENA[base], "variante": "cpu" if a.stem.endswith("_cpu") else "gpu" if a.stem.endswith("_gpu") else "unica",
                       "arquivo": a.name, "sha256": sha256(a), **dados,
                       "pixels_totais": int(dados["res_x"] * pct) * int(dados["res_y"] * pct)})
        print(linhas[-1]["arquivo"], linhas[-1]["amostras"], linhas[-1]["pixels_totais"], flush=True)
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


if __name__ == "__main__":
    main()
