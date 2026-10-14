"""Testes do tratamento do OpenBenchmarking (src/predict/openbenchmarking.py)."""

import numpy as np
import pandas as pd
import pytest
from predict import openbenchmarking as ob


def _envio (**kw) -> dict:
    base = dict(id_resultado="r1", duplicata_de=np.nan, versao_blender_num=4.1, cena="BMW27",
                backend_normalizado="OptiX", valor=10.0, n_rodadas=3, rodadas_cv_pct=0.5,
                tempos_totais="10.5;10.5;10.5", carregamento_medio_s=0.5,
                dispositivo_render="NVIDIA GeForce RTX 3060", gpu_quantidade=1, cpu_modelo="AMD Ryzen 7 5800X",
                cpu_soquetes=1, cpu_nucleos_fisicos_completo=8, cpu_threads_completo=16, cpu_spec_cores=8, so_familia="Linux")
    base.update(kw)
    return base


def test_filtro_versao_falha_e_backend ():
    df = pd.DataFrame([_envio(), _envio(versao_blender_num=2.93), _envio(valor=np.nan),
                       _envio(backend_normalizado="OpenCL"), _envio(duplicata_de="r0")])
    out, funil = ob.filtrar(df)
    assert len(out) == 1
    assert [n for _, n in funil] == sorted([n for _, n in funil], reverse=True)   # o funil só encolhe
    assert out.loc[0, "maquina"] == "NVIDIA GEFORCE RTX 3060|OPTIX"
    assert out.loc[0, "cena"] == "bmw27" and out.loc[0, "versao_blender_minor"] == 1


def test_nucleos_ausentes_vem_da_base_de_especificacoes ():
    out, _ = ob.filtrar(pd.DataFrame([_envio(cpu_nucleos_fisicos_completo=np.nan, cpu_spec_cores=16, cpu_soquetes=2)]))
    assert out.loc[0, "num_cpu_cores"] == 32


def test_classroom_do_pts_nao_e_o_do_opendata ():
    assert ob.CENA["Classroom"] != "classroom" and ob.CENA["Junkshop"] == "junkshop"


def test_carregamento_dedutivo_e_exato ():
    df, _ = ob.filtrar(pd.DataFrame([_envio(n_rodadas=0, carregamento_medio_s=np.nan, tempos_totais="10.75")]))
    out = ob.carregamento(df)
    assert out.loc[0, "carregamento_origem"] == "dedutivo"
    assert out.loc[0, "carregamento_s"] == pytest.approx(0.75)


def test_carregamento_impossivel_sai_e_atipico_fica ():
    linhas = [_envio(carregamento_medio_s=0.5 + 0.01 * i) for i in range(20)]
    linhas += [_envio(carregamento_medio_s=-0.2), _envio(carregamento_medio_s=6.0)]
    out = ob.carregamento(ob.filtrar(pd.DataFrame(linhas))[0])
    assert out.loc[20, "carregamento_origem"] == "descartado" and np.isnan(out.loc[20, "carregamento_s"])
    assert out.loc[21, "carregamento_origem"] == "observado" and bool(out.loc[21, "carregamento_atipico"])


def test_imputacao_usa_so_doadores_medidos ():
    linhas = [_envio(carregamento_medio_s=0.4)] * 6 + [_envio(carregamento_medio_s=np.nan, n_rodadas=3)]
    out = ob.imputar_carregamento(ob.carregamento(ob.filtrar(pd.DataFrame(linhas))[0]))
    assert out.loc[6, "carregamento_origem"] == "imputado_celula_0"
    assert out.loc[6, "carregamento_s"] == pytest.approx(0.4)


def test_vazao_limite_inferior_e_alvo_custo_fixo_vazio ():
    df = ob.imputar_carregamento(ob.carregamento(ob.filtrar(pd.DataFrame([_envio()]))[0]))
    fracoes = pd.Series([0.05], index=pd.MultiIndex.from_tuples([("OPTIX", 4)]))
    cenas = pd.DataFrame([dict(cena_pts="BMW27", variante="gpu", amostras=1225, pixels_totais=518400)])
    v = ob.vazao(df, fracoes, cenas)
    assert v.loc[0, "vazao_limite_inferior"] == pytest.approx(1225 * 518400 / 10.0)
    assert v.loc[0, "vazao_amostras_pixel_s"] == pytest.approx(1225 * 518400 / 9.5)
    ag = ob.agregar(v)
    assert ag["custo_fixo_s"].isna().all() and ag["log_custo_fixo"].isna().all()   # nunca imputado no alvo
    assert (ag["log_vazao_limite_inferior"] <= ag["log_vazao"]).all()
