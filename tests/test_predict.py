"""
Testes do módulo preditivo (src/predict/). Não precisam do Blender nem dos dados
reais: os modelos são treinados em dados sintéticos.
"""

from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import pytest
from predict import dataset, esquema, features_cena, incerteza, metricas, modelos, realimentacao, validacao
from predict.hardware import features_hardware, linha_maquina, nome_dispositivo

# ------------------------------------------------------------------ hardware --

@pytest.mark.parametrize("nome, tipo, esperado", [
    ("NVIDIA GeForce RTX 3060 Laptop GPU", "OPTIX", ("NVIDIA", "GEFORCE_RTX", 2020, 60, True)),
    ("NVIDIA GeForce GTX 1650", "CUDA", ("NVIDIA", "GEFORCE_GTX", 2019, 50, False)),
    ("GeForce GTX 960M", "CUDA", ("NVIDIA", "GEFORCE_GTX", 2014, 60, True)),
    ("Quadro RTX 6000", "OPTIX", ("NVIDIA", "NVIDIA_PRO", 2018, 6, False)),
    ("NVIDIA RTX A2000 12GB", "OPTIX", ("NVIDIA", "NVIDIA_PRO", 2020, 2, False)),
    ("NVIDIA RTX 4000 Ada Generation", "OPTIX", ("NVIDIA", "NVIDIA_PRO", 2022, 4, False)),
    ("Tesla T4", "CUDA", ("NVIDIA", "NVIDIA_PRO", 2018, 4, False)),
    ("NVIDIA TITAN X (Pascal)", "OPTIX", ("NVIDIA", "TITAN", 2016, 99, False)),
    ("AMD Radeon RX 6800M", "HIP", ("AMD", "RADEON_RX", 2020, 8, True)),
    ("AMD Radeon RX 580", "HIP", ("AMD", "RADEON_RX", 2017, 8, False)),
    ("AMD Radeon Pro W7900 Dual Slot", "HIP", ("AMD", "RADEON_PRO", 2023, 9, False)),
    ("AMD Radeon 780M", "HIP", ("AMD", "RADEON_IGPU", 2023, 8, True)),
    ("Apple M2 Max (GPU - 38 cores)", "METAL", ("APPLE", "APPLE_M", 2022, 3, None)),
    ("Intel(R) Arc(TM) A770 Graphics", "ONEAPI", ("INTEL", "ARC", 2022, 7, False)),
    ("11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz", "CPU", ("INTEL", "CORE_I", 2021, 5, True)),
    ("Intel Core i7-1065G7 CPU @ 1.30GHz", "CPU", ("INTEL", "CORE_I", 2020, 7, True)),
    ("Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz", "CPU", ("INTEL", "CORE_I", 2012, 7, False)),
    ("Intel Core Ultra 7 155H", "CPU", ("INTEL", "CORE_ULTRA", 2023, 7, True)),
    ("Intel Xeon Gold 6226 CPU @ 2.70GHz", "CPU", ("INTEL", "XEON_SCALABLE", 2019, 3, None)),
    ("AMD Ryzen 5 5600X 6-Core Processor", "CPU", ("AMD", "RYZEN", 2020, 5, False)),
    ("AMD EPYC 7R13 Processor", "CPU", ("AMD", "EPYC", 2021, 9, None)),
    ("AMD Ryzen Threadripper PRO 7965WX 24-Cores", "CPU", ("AMD", "THREADRIPPER", 2023, 9, None)),
])
def test_hardware_reconhece_nomes_reais (nome, tipo, esperado):
    f = features_hardware(nome, tipo)
    assert (f["hw_fabricante"], f["hw_familia"], f["hw_ano"], f["hw_classe"], f["hw_notebook"]) == esperado


@pytest.mark.parametrize("nome", [None, float("nan"), "", "   ", "?", "Genuine Intel CPU 0000", "Virtual CPU"])
def test_hardware_na_duvida_devolve_none (nome):
    f = features_hardware(nome, "CPU")
    assert f["hw_ano"] is None and f["hw_classe"] is None


def test_apple_conta_nucleos_e_nao_vira_intel ():
    f = features_hardware("Apple M3 Pro (GPU - 18 cores)", "METAL")
    assert f["hw_fabricante"] == "APPLE" and f["hw_nucleos_gpu"] == 18


FICHA_NOTEBOOK = {"sistema": "Windows", "cpu": "11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz", "threads": 12,
                  "nucleos": 6, "gpus": ["Intel(R) UHD Graphics", "NVIDIA GeForce GTX 1650, 4096 MiB"]}


def test_nome_do_dispositivo_segue_o_backend ():
    assert nome_dispositivo("OPTIX", FICHA_NOTEBOOK) == "NVIDIA GeForce GTX 1650"   # não a Intel integrada
    assert nome_dispositivo("CPU", FICHA_NOTEBOOK).startswith("11th Gen")
    assert nome_dispositivo("METAL", {"cpu": "Apple M2 Max"}) == "Apple M2 Max"
    assert nome_dispositivo("HIP", {"cpu": "x", "gpus": []}) is None


def test_linha_maquina_tem_as_colunas_do_esquema ():
    linha = linha_maquina("OPTIX", FICHA_NOTEBOOK, "5.1", "nb")
    assert set(esquema.MAQUINA + esquema.HOST) <= set(linha)
    assert linha["maquina"] == "NVIDIA GEFORCE GTX 1650|OPTIX"
    assert (linha["hw_familia"], linha["host_familia"], linha["versao_blender_major"]) == ("GEFORCE_GTX", "CORE_I", 5)
    assert linha_maquina("HIP", {"cpu": "x"}, "5.1", "pc7")["maquina"] == "TRABALHADOR pc7|HIP"

# ------------------------------------------------------------------- esquema --

@pytest.mark.parametrize("alvo", list(esquema.ESQUEMAS))
def test_nenhum_esquema_tem_coluna_proibida (alvo):
    cols = set(esquema.colunas(alvo))
    assert not cols & esquema.PROIBIDAS_SEMPRE
    assert "cena" not in cols and "cena_sha256" not in cols          # a cena entra pelos atributos (§2.4)
    if alvo == "vazao":
        assert not cols & set(esquema.NA_FORMULA)


@pytest.mark.parametrize("intrusa", ["amostras", "cena"])
def test_esquema_com_coluna_proibida_e_recusado (monkeypatch, intrusa):
    monkeypatch.setitem(esquema.ESQUEMAS, "vazao", esquema.VAZAO + (intrusa,))
    with pytest.raises(ValueError, match=intrusa):
        esquema.colunas("vazao")


def test_montar_x_categoria_nova_vira_nan_e_booleano_vira_numero ():
    treino = pd.DataFrame({"dispositivo_tipo": ["CPU", "OPTIX"], "usa_hdri": ["True", "False"]})
    _, cats = esquema.montar_X(treino, ("dispositivo_tipo", "usa_hdri"))
    novo = pd.DataFrame({"dispositivo_tipo": ["METAL"], "usa_hdri": [True]})
    X, _ = esquema.montar_X(novo, ("dispositivo_tipo", "usa_hdri"), cats)
    assert pd.isna(X["dispositivo_tipo"].iloc[0])
    assert list(X["dispositivo_tipo"].cat.categories) == ["CPU", "OPTIX"]
    assert X["usa_hdri"].iloc[0] == 1.0


def test_montar_x_reclama_de_coluna_ausente ():
    with pytest.raises(ValueError, match="hw_ano"):
        esquema.montar_X(pd.DataFrame({"etapa": ["carregar"]}), ("etapa", "hw_ano"))

# ------------------------------------------------------ métricas e validação --

def test_metricas_previsao_perfeita_e_fator_dois ():
    y = np.log(np.array([10.0, 20.0, 40.0]))
    assert metricas.resumo(y, y)["rmse_log"] == 0
    r = metricas.resumo(y, y + np.log(2))
    assert r["fator_mediano"] == pytest.approx(2.0)
    assert r["erro_rel_mediano"] == pytest.approx(1.0)
    assert r["vies_log"] > 0


def test_teste_blindado_nao_depende_das_outras_maquinas ():
    a = pd.DataFrame({"maquina": [f"m{i}" for i in range(500)]})
    b = pd.concat([a, pd.DataFrame({"maquina": [f"novo{i}" for i in range(300)]})], ignore_index=True)
    _, te_a = validacao.teste_blindado(a)
    _, te_b = validacao.teste_blindado(b)
    assert set(a["maquina"].iloc[te_a]) == set(b["maquina"].iloc[te_b]) & set(a["maquina"])
    assert 0.12 < len(te_a) / len(a) < 0.28


def test_grupo_nunca_fica_dos_dois_lados ():
    grupos = np.repeat([f"m{i}" for i in range(30)], 3)
    for tr, va in validacao.dobras(grupos, "maquina", 5, semente=1):
        assert not set(grupos[tr]) & set(grupos[va])
    dobras_cena = validacao.dobras(np.repeat(["a", "b", "c"], 4), "cena")
    assert len(dobras_cena) == 3 and all(len(set(np.repeat(["a", "b", "c"], 4)[va])) == 1 for _, va in dobras_cena)


def test_mediana_por_grupo_recua_para_o_grupo_menor ():
    X = pd.DataFrame({"dispositivo_tipo": ["CPU", "CPU", "OPTIX"], "hw_familia": ["RYZEN", "CORE_I", "GEFORCE_RTX"]})
    m = modelos.MedianaPorGrupo(("dispositivo_tipo", "hw_familia")).ajustar(X, np.array([1.0, 3.0, 5.0]))
    novo = pd.DataFrame({"dispositivo_tipo": ["CPU", "CPU", "METAL"], "hw_familia": ["CORE_I", "XEON_W", "APPLE_M"]})
    assert m.prever(novo).tolist() == [3.0, 2.0, 3.0]     # exato; só o tipo; global
    assert modelos.MedianaPorGrupo(()).ajustar(X, np.array([1.0, 3.0, 5.0])).prever(novo).tolist() == [3.0] * 3

# ----------------------------------------------------------------- incerteza --

def test_conformal_garante_a_cobertura ():
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    q_ruim = np.full(4000, -2.0)                             # "quantil 90%" muito baixo
    folga = incerteza.calibrar(q_ruim[:2000], y[:2000], "custo_fixo")
    cobertura = incerteza.cobre(y[2000:], incerteza.aplicar(q_ruim[2000:], folga, "custo_fixo"), "custo_fixo")
    assert cobertura.mean() >= 0.88


def test_quantil_ponderado_da_peso_igual_a_cada_cena ():
    # Cena A: 1.000 escores 0; cena B: 10 escores 5. Sem peso, o quantil 90% seria 0.
    s = np.r_[np.zeros(1000), np.full(10, 5.0)]
    w = np.r_[np.full(1000, 1 / 1000), np.full(10, 1 / 10)]
    assert incerteza._quantil_ponderado(s, w, 0.9) == 5.0

# ------------------------------------------------- extração e realimentação --

def test_intervalos_de_frames ():
    assert features_cena.intervalos([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10"
    assert features_cena.intervalos([5]) == "5"
    assert len(features_cena.intervalos(range(1, 10001))) < 10


def _relatorio (**kw) -> dict:
    base = dict(job_id="j1", trabalhador_id="pc1", status="concluido", cena_id="c", frame=1, dispositivo="CPU",
                sistema="Windows", versao_blender="5.1", tempo_previsto_s=10.0, tempo_total_s=12.0,
                tempo_preparacao_s=2.0, tempo_blender_s=11.0, amostra_final=64, amostras_total=64,
                instante_ultima_amostra_s=10.5, lote_id="L1")
    base.update(kw)
    return base


def test_realimentacao_separa_os_tres_relogios_e_explica_o_descarte ():
    rel = pd.DataFrame([_relatorio(), _relatorio(status="abortado"), _relatorio(amostra_final=30),
                        _relatorio(tempo_blender_s=None), _relatorio(tempo_blender_s=12.5)])
    d = realimentacao.separar_tempos(rel)
    assert d.loc[0, "descarte"] == ""
    assert d.loc[0, "amostragem_s"] == pytest.approx(8.5)            # 10,5 - 2,0
    assert d.loc[0, "carregar_s"] == pytest.approx(1.0)              # 12,0 - 11,0
    assert d.loc[0, "preparar_finalizar_s"] == pytest.approx(2.5)    # 11,0 - 8,5
    assert d["descarte"].tolist()[1:] == ["abortado", "amostragem incompleta", "sem relógio do log", "tempo impossível"]


def test_dataset_nao_treina_com_carregar_imputado ():
    pub = pd.DataFrame({
        "fonte": ["openbenchmarking", "openbenchmarking", "opendata"],
        "carregamento_origem": ["medido", "imputado", "imputado_ob_1"],
        "carregamento_s": [0.5, 0.7, 0.9], "log_custo_fixo": [np.nan, np.nan, np.log(3.0)],
    })
    prod = pd.DataFrame({"log_carregar": [np.log(1.2)], "log_preparar": [np.log(2.0)]})
    c = dataset.montar_custo(pub, prod)
    carregar = c[c["etapa"] == "carregar"]["log_custo"]
    assert sorted(np.round(np.exp(carregar), 6)) == [0.5, 1.2]       # o 0,7 e o 0,9 imputados ficam fora
    assert (c["etapa"] == "preparar_finalizar").sum() == 2           # Open Data + produção

# ------------------------------------------------- artefato, preditor, mestre --

def _dataset_sintetico (n_cenas=10, n_maquinas=60, rng=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dados de mentira com uma regra conhecida:
      log vazão = 10 + 0,1 (ano - 2015) + 1,0 [GPU] - 0,3 log_triangulos
      log custo = 0,02 texturas - 1,0 [etapa carregar] - 0,02 (ano - 2015)
    """
    rng = rng or np.random.default_rng(1)
    maquinas = []
    for i in range(n_maquinas):
        gpu = i % 2 == 1
        ano = int(rng.integers(2014, 2025))
        maquinas.append({
            "maquina": f"MAQ{i}|{'OPTIX' if gpu else 'CPU'}", "dispositivo_tipo": "OPTIX" if gpu else "CPU",
            "num_cpu_cores": 8.0, "num_cpu_threads": 16.0, "num_cpu_sockets": 1.0, "sistema": "Windows",
            "versao_blender_major": 5.0, "versao_blender_minor": 1.0,
            "hw_fabricante": "NVIDIA" if gpu else "INTEL", "hw_familia": "GEFORCE_RTX" if gpu else "CORE_I",
            "hw_ano": float(ano), "hw_classe": 6.0, "hw_notebook": 0.0, "hw_nucleos_gpu": np.nan, "hw_vram_gb": np.nan,
            "host_familia": "CORE_I", "host_ano": float(ano), "host_classe": 5.0,
        })
    maquinas = pd.DataFrame(maquinas)
    vazao, custo = [], []
    for c in range(n_cenas):
        cena = {col: 0.0 for col in esquema.COLUNAS_CENA}
        cena.update(log_triangulos=rng.uniform(8, 16), num_texturas=float(rng.integers(0, 50)),
                    cycles_denoiser="OPENIMAGEDENOISE", amostras=128.0, pixels_totais=640.0 * 360,
                    cena=f"cena{c:02d}", cena_sha256=f"{c:064x}")
        for _, m in maquinas.iterrows():
            base = {**cena, **m.to_dict(), "fonte": "opendata", "peso": 1.0}
            lv = (10 + 0.1 * (m["hw_ano"] - 2015) + (1.0 if m["dispositivo_tipo"] == "OPTIX" else 0.0)
                  - 0.3 * cena["log_triangulos"] + rng.normal(0, 0.05))
            vazao.append({**base, "log_vazao": lv})
            for etapa in esquema.ETAPAS:
                lc = (0.02 * cena["num_texturas"] - (1.0 if etapa == "carregar" else 0.0)
                      - 0.02 * (m["hw_ano"] - 2015) + rng.normal(0, 0.05))
                custo.append({**base, "etapa": etapa, "log_custo": lc})
    return pd.DataFrame(vazao), pd.DataFrame(custo), maquinas


@pytest.fixture(scope="module")
def artefato (tmp_path_factory):
    from predict.treinar import treinar_alvo
    pasta = tmp_path_factory.mktemp("modelo")
    dv, dc, maquinas = _dataset_sintetico()
    params = {"min_child_samples": 5, "n_estimators": 300}
    meta = {"etapas_custo": list(esquema.ETAPAS),
            "alvos": {"vazao": treinar_alvo(dv, "vazao", params, pasta),
                      "custo_fixo": treinar_alvo(dc, "custo_fixo", params, pasta)}}
    (pasta / "metadados.json").write_text(json.dumps(meta), encoding="utf-8")
    return pasta, dv, maquinas


def _frames (dv: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Como o extrator entrega: só colunas de cena, uma linha por frame."""
    return dv.drop_duplicates("cena_sha256").head(n)[list(esquema.COLUNAS_CENA)].reset_index(drop=True)


def test_preditor_ida_e_volta_e_formula (artefato):
    from predict.preditor import Preditor
    pasta, dv, maquinas = artefato
    p = Preditor(pasta)
    frames = _frames(dv)
    lenta = maquinas[(maquinas["dispositivo_tipo"] == "CPU")].nsmallest(1, "hw_ano")
    rapida = maquinas[(maquinas["dispositivo_tipo"] == "OPTIX")].nlargest(1, "hw_ano")
    linhas = pd.concat([lenta, rapida], ignore_index=True)
    P = p.prever_matriz_p(frames, linhas)
    assert P.shape == (5, 2) and np.isfinite(P).all() and (P > 0).all()
    assert (P[:, 0] > P[:, 1]).all()                          # a lenta é mais lenta em todos os frames
    # Frames com colunas de máquina (de outra máquina!) não podem mudar o resultado.
    sujos = frames.assign(hw_ano=1990.0, dispositivo_tipo="CPU", versao_blender_major=2.0)
    assert np.allclose(p.prever_matriz_p(sujos, linhas), P)
    # A célula (i, j) é a fórmula aplicada ao par (frame i, máquina j), com as duas etapas somadas.
    par = frames.iloc[[3]].merge(linhas.iloc[[1]], how="cross")
    v = np.exp(p._prever("vazao", par, False))[0]
    c = sum(np.exp(p._prever("custo_fixo", par.assign(etapa=e), False))[0] for e in esquema.ETAPAS)
    assert P[3, 1] == pytest.approx(c + frames["amostras"][3] * frames["pixels_totais"][3] / v)


def test_cena_nova_ganha_limite_mais_pessimista (artefato):
    from predict.preditor import Preditor
    pasta, dv, maquinas = artefato
    p = Preditor(pasta)
    frames, linhas = _frames(dv, 3), maquinas.head(4)
    conhecida = dv["cena_sha256"].iloc[0]
    assert p.cena_conhecida("vazao", conhecida) and not p.cena_conhecida("vazao", "f" * 64)
    P_conhecida = p.prever_matriz_p(frames, linhas, conhecida, conservador=True)
    P_nova = p.prever_matriz_p(frames, linhas, "f" * 64, conservador=True)
    assert (P_nova >= P_conhecida * (1 - 1e-9)).all()
    assert np.allclose(p.prever_matriz_p(frames, linhas, conhecida), p.prever_matriz_p(frames, linhas, "f" * 64))


@pytest.mark.slow
def test_inferencia_de_um_lote_grande_e_rapida (artefato):
    """RNF-02: 10.000 frames x 10 máquinas em menos de 5 s."""
    from predict.preditor import Preditor
    pasta, dv, maquinas = artefato
    p = Preditor(pasta)
    frames = _frames(dv, 10).sample(10_000, replace=True, random_state=0)
    inicio = time.perf_counter()
    P = p.prever_matriz_p(frames, maquinas.head(10))
    assert P.shape == (10_000, 10)
    assert time.perf_counter() - inicio < 5.0


def test_planner_usa_a_ia_e_mantem_inelegivel (artefato, monkeypatch):
    import comum.util
    import master.planner as planner
    from predict.preditor import Preditor
    pasta, dv, _ = artefato
    monkeypatch.setattr(features_cena, "extrair", lambda *a, **k: _frames(dv, 3))
    monkeypatch.setattr(comum.util, "sha256_arquivo", lambda *a, **k: "f" * 64)
    monkeypatch.setattr(planner, "_ia", planner.IA(Preditor(pasta), {"5.1": "blender"}, pasta, False))
    ficha_gpu = {"sistema": "Windows", "cpu": "AMD Ryzen 9 7950X 16-Core Processor", "threads": 32, "nucleos": 16,
                 "gpus": ["NVIDIA GeForce RTX 4090, 24564 MiB"]}
    ficha_cpu = {"sistema": "Windows", "cpu": "Intel(R) Core(TM) i5-4460 CPU @ 3.20GHz", "threads": 4, "nucleos": 4}
    maquinas = [
        planner.Maquina("rapida", ficha_gpu, ["5.1"], "OPTIX"),
        planner.Maquina("lenta", ficha_cpu, ["5.1"], "CPU"),
        planner.Maquina("desconhecida", {"sistema": "Linux", "cpu": "?", "threads": 8}, ["5.1"], "HIP"),
        planner.Maquina("antiga", ficha_cpu, ["4.2"], "CPU"),
    ]
    P = planner.estimar_matriz_p(planner.Cena("c", "c.blend", "5.1"), [1, 2, 3], maquinas)
    assert np.isfinite(P[:, :3]).all() and (P[:, :3] > 0).all()   # nome desconhecido também é previsto
    assert np.isinf(P[:, 3]).all()                                # Blender 4.2 não abre cena 5.1

# ---------------------------------------------------------- tres_matrizes --

def test_ordem_de_chegada_maquina_livre_pega_o_proximo ():
    from predict.tres_matrizes import makespan, ordem_de_chegada
    # Máquina 0 é 4x mais rápida: termina cada frame antes e pega mais deles
    P = np.array([[1.0, 4.0]] * 5)
    plano = ordem_de_chegada(P)
    assert plano == [[0, 2, 3, 4], [1]]
    assert makespan(plano, P) == 4.0
    # Ignora o custo: o frame caro vai para quem estiver livre, mesmo sendo a máquina lenta
    P = np.array([[1.0, 1.0], [1.0, 9.0]])
    assert ordem_de_chegada(P) == [[0], [1]]
