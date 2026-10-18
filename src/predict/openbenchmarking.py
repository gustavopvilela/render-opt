from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from predict.hardware import normalizar
from predict.opendata import _moda, acrescentar_hardware

RAIZ = Path(__file__).resolve().parents[2]
ENVIOS = RAIZ / "data" / "raw" / "openbenchmarking_blender_envios_completa_especificacoes.csv"
CENAS_CSV = RAIZ / "data" / "raw" / "cenas_openbenchmarking.csv"
OPENDATA_CSV = RAIZ / "data" / "processed" / "dataset_opendata_sem_agregar.csv"
OPENDATA = RAIZ / "data" / "processed" / "opendata.parquet"
SAIDA_OB = RAIZ / "data" / "processed" / "openbenchmarking.parquet"
SAIDA_FINAL = RAIZ / "data" / "processed" / "medicoes_publicas.parquet"

VERSAO_MINIMA = 3.0
TIPO = {"CPU-Only": "CPU", "CUDA": "CUDA", "OptiX": "OPTIX", "HIP": "HIP", "oneAPI": "ONEAPI", "Metal": "METAL"}
CENA = {"Junkshop": "junkshop", "Classroom": "classroom_pts", "BMW27": "bmw27", "Fishy Cat": "fishy_cat",
        "Pabellon Barcelona": "pabellon", "Barbershop": "barbershop"}
SISTEMA = {"Linux": "Linux", "Windows": "Windows", "macOS": "Darwin"}
CV_MAX_PCT = 10.0
Z_MAX = 3.5
MIN_DOADORES = 5
SENTINELA_AMOSTRAS = 2 ** 24

COLUNAS = [
    "id_resultado", "duplicata_de", "versao_blender_num", "cena", "backend_normalizado", "valor",
    "n_rodadas", "rodadas_cv_pct", "tempos_totais", "carregamento_medio_s",
    "dispositivo_render", "gpu_quantidade", "cpu_modelo", "cpu_soquetes",
    "cpu_nucleos_fisicos_completo", "cpu_threads_completo", "cpu_spec_cores", "so_familia",
]


def carregar (caminho: Path = ENVIOS) -> pd.DataFrame:
    return pd.read_csv(caminho, usecols=COLUNAS, low_memory=False)


def filtrar (df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    funil = [("linhas lidas", len(df))]

    def etapa (nome: str, manter: pd.Series) -> None:
        nonlocal df
        df = df[manter.fillna(False)]
        funil.append((nome, len(df)))

    etapa("sem duplicatas", df["duplicata_de"].isna())
    etapa(f"Blender >= {VERSAO_MINIMA}", df["versao_blender_num"] >= VERSAO_MINIMA)
    etapa("com resultado (teste não falhou)", df["valor"] > 0)
    etapa("backend com par no Open Data", df["backend_normalizado"].isin(TIPO))
    etapa("dispositivo identificado", df["dispositivo_render"].notna())
    tipo = df["backend_normalizado"].map(TIPO)
    etapa("uma GPU só (backends de GPU)", (tipo == "CPU") | (df["gpu_quantidade"].fillna(1) == 1))
    etapa(f"repetições estáveis (CV <= {CV_MAX_PCT:.0f}%)", df["rodadas_cv_pct"].isna() | (df["rodadas_cv_pct"] <= CV_MAX_PCT))

    df = df.copy()
    df["dispositivo_tipo"] = df["backend_normalizado"].map(TIPO)
    df["cena"] = df["cena"].map(CENA)
    df["versao_blender_major"] = df["versao_blender_num"].astype(int)
    df["versao_blender_minor"] = (df["versao_blender_num"] * 10).round().astype(int) % 10
    df["maquina"] = df["dispositivo_render"].map(normalizar) + "|" + df["dispositivo_tipo"]
    df["num_cpu_cores"] = df["cpu_nucleos_fisicos_completo"].fillna(df["cpu_spec_cores"] * df["cpu_soquetes"])
    return df.reset_index(drop=True), funil


def carregamento (df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["carregamento_medio_s"].astype(float)
    origem = pd.Series(np.where(c.notna(), "observado", "ausente"), index=df.index, dtype=object)

    tempos = df["tempos_totais"].astype("string").str.split(";")
    um_tempo = tempos.str.len() == 1
    dedutivel = c.isna() & (df["n_rodadas"].fillna(0) == 0) & um_tempo.fillna(False)
    c[dedutivel] = pd.to_numeric(tempos[dedutivel].str[0], errors="coerce") - df.loc[dedutivel, "valor"]
    origem[dedutivel] = "dedutivo"

    invalido = c.notna() & ((c <= 0) | (c >= df["valor"]))
    lc = np.log(c.where(~invalido))
    grupo = [df["cena"], df["dispositivo_tipo"]]
    mediana = lc.groupby(grupo).transform("median")
    mad = (lc - mediana).abs().groupby(grupo).transform("median")
    z = 0.6745 * (lc - mediana) / mad
    c[invalido] = np.nan
    origem[invalido] = "descartado"

    df["carregamento_s"], df["carregamento_origem"] = c, origem
    df["carregamento_atipico"] = z.abs() > Z_MAX
    return df


def imputar_carregamento (df: pd.DataFrame) -> pd.DataFrame:
    return _imputar_celula(df, "carregamento_s", "carregamento_origem",
                           [["cena", "dispositivo_tipo", "versao_blender_major"], ["cena", "dispositivo_tipo"],
                            ["dispositivo_tipo"]])


def _imputar_celula (df: pd.DataFrame, col: str, col_origem: str, celulas: list[list[str]],
                     doadores: pd.DataFrame | None = None, rotulo: str = "imputado_celula") -> pd.DataFrame:
    df = df.copy()
    doadores = df if doadores is None else doadores
    obs = doadores[doadores[col].notna() & doadores[col_origem].isin(["observado", "dedutivo"])]
    for nivel, chave in enumerate(celulas):
        g = np.log(obs[col]).groupby([obs[k] for k in chave])
        med = g.median()[g.size() >= MIN_DOADORES]
        falta = df[col].isna()
        if not falta.any() or med.empty:
            continue
        idx = pd.MultiIndex.from_frame(df.loc[falta, chave]) if len(chave) > 1 else df.loc[falta, chave[0]]
        valor = np.exp(pd.Series(med.reindex(idx).to_numpy(), index=df.index[falta]))
        achou = valor.notna()
        df.loc[valor.index[achou], col] = valor[achou]
        df.loc[valor.index[achou], col_origem] = f"{rotulo}_{nivel}"
    return df


def fracoes_opendata (caminho: Path = OPENDATA_CSV) -> pd.Series:
    od = pd.read_csv(caminho, usecols=["dispositivo_tipo", "versao_blender_major", "custo_fixo_s",
                                       "medido_total_render_time", "medido_num_amostras"])
    od = od[(od["medido_num_amostras"] < SENTINELA_AMOSTRAS) & (od["custo_fixo_s"] > 0)]
    frac = od["custo_fixo_s"] / od["medido_total_render_time"]
    return frac.groupby([od["dispositivo_tipo"], od["versao_blender_major"]]).median().rename("fracao")


def vazao (df: pd.DataFrame, fracoes: pd.Series, cenas: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cenas = cenas.assign(cena=cenas["cena_pts"].map(CENA), trabalho=cenas["amostras"] * cenas["pixels_totais"])
    variante = np.where(df["dispositivo_tipo"] == "CPU", "cpu", "gpu")
    variante = np.where(df["cena"] == "junkshop", "unica", variante)
    trabalho = cenas.set_index(["cena", "variante"])["trabalho"]
    df["amostras_x_pixels"] = trabalho.reindex(pd.MultiIndex.from_arrays([df["cena"], variante])).to_numpy()

    f = fracoes.reindex(pd.MultiIndex.from_frame(df[["dispositivo_tipo", "versao_blender_major"]])).to_numpy()
    df["custo_fixo_imputado_s"] = f * df["valor"]
    df["tempo_render_s"] = df["valor"]
    df["vazao_amostras_pixel_s"] = df["amostras_x_pixels"] / (df["valor"] - df["custo_fixo_imputado_s"])
    df["vazao_limite_inferior"] = df["amostras_x_pixels"] / df["valor"]
    return df


def agregar (df: pd.DataFrame) -> pd.DataFrame:
    chave = ["maquina", "cena", "versao_blender_major"]
    medido = df["carregamento_origem"].isin(["observado", "dedutivo"])
    df = df.assign(_carreg_medido=df["carregamento_s"].where(medido),
                   log_vazao=np.log(df["vazao_amostras_pixel_s"]),
                   log_vazao_limite_inferior=np.log(df["vazao_limite_inferior"]),
                   log_tempo_render=np.log(df["tempo_render_s"]))
    g = df.groupby(chave, sort=False)
    out = g.agg(
        n_medicoes=("valor", "size"),
        log_vazao=("log_vazao", "median"),
        log_vazao_limite_inferior=("log_vazao_limite_inferior", "median"),
        log_tempo_render=("log_tempo_render", "median"),
        custo_fixo_imputado_s=("custo_fixo_imputado_s", "median"),
        carregamento_medido=("_carreg_medido", "median"),
        carregamento_qualquer=("carregamento_s", "median"),
        dispositivo_nome=("dispositivo_render", "first"),
        dispositivo_tipo=("dispositivo_tipo", "first"),
        num_cpu_cores=("num_cpu_cores", "median"),
        num_cpu_threads=("cpu_threads_completo", "median"),
        num_cpu_sockets=("cpu_soquetes", "median"),
        versao_blender_minor=("versao_blender_minor", "median"),
    ).reset_index()
    out["cpu_nome"] = g["cpu_modelo"].agg(_moda).values
    out["sistema"] = g["so_familia"].agg(_moda).map(SISTEMA).values

    out["carregamento_s"] = out["carregamento_medido"].fillna(out["carregamento_qualquer"])
    out["carregamento_origem"] = np.where(out["carregamento_medido"].notna(), "medido",
                                          np.where(out["carregamento_s"].notna(), "imputado", "ausente"))
    out = out.drop(columns=["carregamento_medido", "carregamento_qualquer"])

    out["vazao_amostras_pixel_s"] = np.exp(out["log_vazao"])
    out["vazao_origem"] = np.where(out["log_vazao"].notna(), "derivada_custo_imputado", "ausente")
    out["custo_fixo_s"] = np.nan
    out["log_custo_fixo"] = np.nan
    out["custo_fixo_origem"] = "ausente"
    out["fonte"] = "openbenchmarking"
    return out


def juntar (ob: pd.DataFrame, od: pd.DataFrame, envios_ob: pd.DataFrame) -> pd.DataFrame:
    od = od.copy()
    od["maquina"] = od["dispositivo_nome"].map(normalizar) + "|" + od["dispositivo_tipo"]
    od["fonte"] = "opendata"
    od["vazao_origem"] = "medida"
    od["custo_fixo_origem"] = np.where(od["custo_fixo_s"].notna(), "medido", "ausente")
    od["carregamento_s"], od["carregamento_origem"] = np.nan, "ausente"
    od = _imputar_celula(od, "carregamento_s", "carregamento_origem",
                         [["cena", "dispositivo_tipo", "versao_blender_major"], ["dispositivo_tipo", "versao_blender_major"],
                          ["dispositivo_tipo"]],
                         doadores=envios_ob, rotulo="imputado_ob")
    final = pd.concat([od, ob], ignore_index=True, sort=False)
    return final


def diagnostico (envios: pd.DataFrame, ob: pd.DataFrame, od_agregado: pd.DataFrame, opendata_csv: Path = OPENDATA_CSV) -> None:
    od = pd.read_csv(opendata_csv, usecols=["cena", "dispositivo_tipo", "versao_blender_major", "custo_fixo_s",
                                            "medido_total_render_time", "medido_num_amostras"])
    od = od[(od["medido_num_amostras"] < SENTINELA_AMOSTRAS) & (od["custo_fixo_s"] > 0)]
    print("\n[1] custo fixo por razão, cena deixada de fora (Open Data, por submissão)")
    print("    cena        |erro| mediano log c   |erro| mediano log v")
    for cena in sorted(od["cena"].unique()):
        tr, te = od[od["cena"] != cena], od[od["cena"] == cena]
        fr = (tr["custo_fixo_s"] / tr["medido_total_render_time"]).groupby([tr["dispositivo_tipo"], tr["versao_blender_major"]]).median()
        f = fr.reindex(pd.MultiIndex.from_frame(te[["dispositivo_tipo", "versao_blender_major"]])).to_numpy()
        T, c = te["medido_total_render_time"].to_numpy(), te["custo_fixo_s"].to_numpy()
        erro_c = np.abs(np.log(f * T) - np.log(c))
        erro_v = np.abs(np.log((T - c) / (T - f * T)))
        print(f"    {cena:10s}  {np.nanmedian(erro_c):18.3f}   {np.nanmedian(erro_v):18.3f}")

    od_j = od_agregado[od_agregado["cena"] == "junkshop"].assign(
        maquina=lambda d: d["dispositivo_nome"].map(normalizar) + "|" + d["dispositivo_tipo"])
    par = ob[ob["cena"] == "junkshop"].merge(od_j[["maquina", "versao_blender_major", "log_vazao"]],
                                             on=["maquina", "versao_blender_major"], suffixes=("_ob", "_od"))
    print(f"\n[2] ponte junkshop: {len(par)} pares (máquina, versão maior)")
    for col, nome in (("log_vazao_ob", "com custo fixo imputado"), ("log_vazao_limite_inferior", "limite inferior (c = 0)")):
        e = par[col] - par["log_vazao_od"]
        print(f"    {nome:26s} viés {e.median():+.3f}  |erro| mediano {e.abs().median():.3f}  "
              f"correlação {np.corrcoef(par[col], par['log_vazao_od'])[0, 1]:.3f}")
    for tipo, gg in par.groupby("dispositivo_tipo"):
        e = gg["log_vazao_ob"] - gg["log_vazao_od"]
        print(f"      {tipo:7s} n={len(gg):3d}  viés {e.median():+.3f}")

    obs = envios[envios["carregamento_origem"].isin(["observado", "dedutivo"])]
    print("\n[3] carregamento imputado sem a cena, cena deixada de fora (OB, por envio)")
    for cena in sorted(obs["cena"].unique()):
        tr, te = obs[obs["cena"] != cena], obs[obs["cena"] == cena]
        med = np.log(tr["carregamento_s"]).groupby([tr["dispositivo_tipo"], tr["versao_blender_major"]]).median()
        p = med.reindex(pd.MultiIndex.from_frame(te[["dispositivo_tipo", "versao_blender_major"]])).to_numpy()
        e = p - np.log(te["carregamento_s"].to_numpy())
        print(f"    {cena:14s} n={len(te):5d}  viés {np.nanmedian(e):+.3f}  |erro| mediano {np.nanmedian(np.abs(e)):.3f}")


def main () -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envios", type=Path, default=ENVIOS)
    ap.add_argument("--cenas", type=Path, default=CENAS_CSV)
    ap.add_argument("--opendata", type=Path, default=OPENDATA)
    ap.add_argument("--opendata-csv", type=Path, default=OPENDATA_CSV)
    ap.add_argument("--diagnostico", action="store_true")
    args = ap.parse_args()

    envios, funil = filtrar(carregar(args.envios))
    print("funil:")
    for nome, n in funil:
        print(f"  {nome:40s} {n:7,d}")

    envios = carregamento(envios)
    print("\ncarregamento antes de imputar:", envios["carregamento_origem"].value_counts().to_dict())
    envios = imputar_carregamento(envios)
    print("carregamento depois de imputar:", envios["carregamento_origem"].value_counts().to_dict())
    print(f"carregamento atípico (marcado, mantido): {int(envios['carregamento_atipico'].sum())}")
    envios = vazao(envios, fracoes_opendata(args.opendata_csv), pd.read_csv(args.cenas))
    print(f"vazão calculada: {envios['vazao_amostras_pixel_s'].notna().mean():.1%} das linhas")

    ob = acrescentar_hardware(agregar(envios))
    SAIDA_OB.parent.mkdir(parents=True, exist_ok=True)
    ob.to_parquet(SAIDA_OB, index=False)
    print(f"\nOB agregado: {len(ob):,} linhas, {ob['maquina'].nunique():,} máquinas")
    print("por cena:", ob["cena"].value_counts().to_dict())
    print("por tipo:", ob["dispositivo_tipo"].value_counts().to_dict())
    print("por versão:", ob["versao_blender_major"].value_counts().sort_index().to_dict())
    print(f"hw_ano preenchido: {ob['hw_ano'].notna().mean():.1%}")

    od = pd.read_parquet(args.opendata)
    final = juntar(ob, od, envios)
    final.to_parquet(SAIDA_FINAL, index=False)
    comuns = set(od["dispositivo_nome"].map(normalizar) + "|" + od["dispositivo_tipo"]) & set(ob["maquina"])
    print(f"\nfinal: {len(final):,} linhas, {final['maquina'].nunique():,} máquinas "
          f"({len(comuns):,} presentes nas duas fontes)")
    print("linhas com log_vazao:", final.groupby("fonte")["log_vazao"].apply(lambda s: int(s.notna().sum())).to_dict())
    print("linhas com log_custo_fixo:", final.groupby("fonte")["log_custo_fixo"].apply(lambda s: int(s.notna().sum())).to_dict())
    print("carregamento no Open Data:", final.loc[final["fonte"] == "opendata", "carregamento_origem"].value_counts().to_dict())
    print(f"escrito: {SAIDA_OB}\n         {SAIDA_FINAL}")

    if args.diagnostico:
        diagnostico(envios, ob, od, args.opendata_csv)


if __name__ == "__main__":
    main()
