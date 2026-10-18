import csv, json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans", "Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": "#9AA0A6",
    "axes.linewidth": 0.8,
    "xtick.color": "#5F6368",
    "ytick.color": "#202124",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "hatch.linewidth": 0.8,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Noto Sans",
    "mathtext.it": "Noto Sans:italic",
})
TEXTO, SECUNDARIO, GRADE, FAIXA = "#202124", "#5F6368", "#E3E5E8", "#F6F7F9"
PALETA = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#EDC948", "#76B7B2",
          "#9C755F", "#FF9DA7", "#A0CBE8", "#FFBE7D", "#8CD17D", "#BAB0AC"]
PERDIDA = "#C0392B"

analise, prefixo = Path(sys.argv[1]), Path(sys.argv[2])
ler = lambda nome: list(csv.DictReader((analise / nome).open(encoding="utf-8")))
plano, execucao = ler("plano.csv"), [l for l in ler("execucao.csv") if l["fim_s"]]
resumo = json.loads((analise / "resumo.json").read_text(encoding="utf-8"))
f = lambda v: float(v) if v not in ("", None) else 0.0

maquinas = sorted({l["maquina"] for l in plano} | {l["maquina"] for l in execucao})
cenas = sorted({l["cena_id"] for l in plano} | {l["cena_id"] for l in execucao})
cor = {c: PALETA[i % len(PALETA)] for i, c in enumerate(cenas)}
y = {m: i for i, m in enumerate(maquinas)}
fim_plano = resumo["makespan_planejado_s"] or 0.0
fim_real = resumo["makespan_real_s"] or max((f(l["fim_s"]) for l in execucao), default=0.0)
limite = max(fim_plano, fim_real) * 1.06

if limite > 2 * 3600:
    unidade, divisor = "h", 3600
elif limite > 10 * 60:
    unidade, divisor = "min", 60
else:
    unidade, divisor = "s", 1

def clarear (c, quanto=0.55):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * quanto, g + (1 - g) * quanto, b + (1 - b) * quanto)

def cor_do_texto (c):
    r, g, b = to_rgb(c)
    return TEXTO if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"

def duracao (s):
    return f"{s / 3600:.2f} h" if s >= 2 * 3600 else f"{s / 60:.1f} min" if s >= 10 * 60 else f"{s:.1f} s"

def figura (titulo):
    fig, eixo = plt.subplots(figsize=(12, 0.62 * len(maquinas) + 2.1))
    for m, i in y.items():
        if i % 2 == 0:
            eixo.axhspan(i - 0.5, i + 0.5, color=FAIXA, zorder=0, linewidth=0)
    eixo.set_xlim(0, limite)
    eixo.set_ylim(len(maquinas) - 0.5, -0.5)
    eixo.set_yticks(range(len(maquinas)), maquinas, fontsize=9.5)
    bruto = limite / divisor / 10
    escala = 10 ** math.floor(math.log10(bruto))
    passo = next(p * escala for p in (1, 2, 2.5, 5, 10) if p * escala >= bruto)
    eixo.xaxis.set_major_locator(MultipleLocator(passo * divisor))
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / divisor:g}"))
    eixo.set_xlabel(f"Tempo desde o início do lote ({unidade})", color=SECUNDARIO, labelpad=6)
    eixo.grid(axis="x", color=GRADE, linewidth=0.7, zorder=1)
    eixo.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        eixo.spines[lado].set_visible(False)
    fig.text(0.012, 0.975, titulo, fontsize=14, fontweight="bold", color=TEXTO, va="top")
    return fig, eixo

def bloco (eixo, m, inicio, preparo, total, cena, rotulo, perdida=False, fora=False):
    base = cor[cena]
    preparo = min(preparo, total)
    if preparo > 0:
        eixo.barh(y[m], preparo, left=inicio, height=0.68, color="white", edgecolor=base, hatch="////", linewidth=0.6, zorder=3)
    if perdida:
        eixo.barh(y[m], total - preparo, left=inicio + preparo, height=0.68, color="white", edgecolor=PERDIDA,
                  hatch="xxx", linewidth=0.9, zorder=3)
    else:
        eixo.barh(y[m], total - preparo, left=inicio + preparo, height=0.68, color=base, edgecolor="white", linewidth=0.6, zorder=3)
    largura_px = (total - preparo) / limite * eixo.get_window_extent().width
    if largura_px > 7 * len(rotulo) + 4:
        eixo.text(inicio + preparo + (total - preparo) / 2, y[m], rotulo, ha="center", va="center", fontsize=7,
                  color=PERDIDA if perdida else cor_do_texto(base), zorder=4,
                  bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none") if perdida else None)
    if fora:
        eixo.scatter(inicio + total / 2, y[m] - 0.41, marker="v", s=26, color=TEXTO, linewidths=0, zorder=5)

def marcar_makespan (eixo, valor):
    eixo.axvline(valor, color=TEXTO, linestyle=(0, (4, 3)), linewidth=1.0, zorder=6)
    direita = valor > 0.8 * limite
    eixo.annotate(r"$\mathit{Makespan}$" + f" = {duracao(valor)}", (valor, -0.5), xytext=(-4 if direita else 4, 2), textcoords="offset points",
                  fontsize=8.5, color=TEXTO, ha="right" if direita else "left", va="bottom", annotation_clip=False,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRADE, linewidth=0.7))

def legenda (fig, extras):
    itens = [Patch(facecolor=cor[c], edgecolor="none", label=c) for c in cenas]
    itens += [Patch(facecolor="white", edgecolor="#7F868C", hatch="////", label="Preparação")] + extras
    fig.legend(handles=itens, loc="lower center", ncol=min(len(itens), 6), frameon=False, fontsize=8.5,
               handlelength=1.4, handleheight=0.9, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))

def salvar (fig, sufixo):
    altura = fig.get_figheight()
    linhas_legenda = -(-(len(cenas) + 3) // 6)
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - 0.75 / altura, bottom=(0.75 + 0.22 * linhas_legenda) / altura)
    for ext in ("png", "pdf"):
        destino = prefixo.with_name(f"{prefixo.name}_{sufixo}.{ext}")
        fig.savefig(destino, dpi=300, facecolor="white")
        print(destino)
    plt.close(fig)

nome = f"{resumo['lote_id']} · {'Simulated Annealing' if resumo['distribuicao'] == 'sa' else 'Flamenco'}"

if plano:
    fig, eixo = figura(f"{nome}: previsto")
    for l in plano:
        bloco(eixo, l["maquina"], f(l["inicio_s"]), f(l["setup_s"]), f(l["fim_s"]) - f(l["inicio_s"]), l["cena_id"], l["frame"])
    marcar_makespan(eixo, fim_plano)
    legenda(fig, [])
    salvar(fig, "previsto")

fig, eixo = figura(f"{nome}: executado")
for l in execucao:
    fora = bool(l["maquina_planejada"]) and l["maquina_planejada"] != l["maquina"] and l["desfecho"] == "concluido"
    bloco(eixo, l["maquina"], f(l["inicio_s"]), f(l["preparacao_s"]), f(l["fim_s"]) - f(l["inicio_s"]), l["cena_id"],
          l["frame"], l["desfecho"] != "concluido", fora)
marcar_makespan(eixo, fim_real)
extras = [Patch(facecolor="white", edgecolor=PERDIDA, hatch="xxx", label="Tentativa perdida")]
if plano:
    extras.append(Line2D([], [], marker="v", linestyle="none", color=TEXTO, markersize=5, label="Fora da máquina planejada"))
legenda(fig, extras)
salvar(fig, "executado")
