from __future__ import annotations
from typing import Optional
import pandas as pd
from predict.especificacoes import DISPOSITIVO, HOSPEDEIRA

CHAVES = ("origem", "fonte", "maquina", "cena", "cena_sha256", "arquivo_blend", "frame",
          "dispositivo_nome", "cpu_nome", "medicao_id", "trabalhador_id", "job_id", "lote_id", "peso", "peso_base", "n_obs")
ALVOS = ("vazao_amostras_pixel_s", "log_vazao", "custo_fixo_s", "log_custo_fixo", "log_custo",
         "carregar_s", "log_carregar", "preparar_finalizar_s", "log_preparar")
POS_RENDER = ("n_medicoes", "medido_num_amostras", "medido_time_for_samples", "medido_total_render_time",
              "medido_samples_per_minute", "medido_pico_memoria_mb", "tempo_total_s", "tempo_amostragem_s",
              "amostragem_s", "tempo_preparacao_s", "tempo_blender_s", "tempo_gravacao_s",
              "instante_ultima_amostra_s", "amostra_final", "amostras_total", "memoria_max_mb",
              "tempo_previsto_s")
NA_FORMULA = ("amostras", "amostras_x_pixels", "log_amostras_x_pixels", "res_x", "res_y", "pixels_totais")

AUXILIARES_OB = ("n_medicoes_custo", "log_vazao_limite_inferior", "log_tempo_render", "tempo_render_s", "custo_fixo_imputado_s",
                 "carregamento_s", "carregamento_atipico", "vazao_origem", "custo_fixo_origem", "carregamento_origem")

PROIBIDAS_SEMPRE = frozenset(CHAVES + ALVOS + POS_RENDER + AUXILIARES_OB)

MAQUINA = (
    "dispositivo_tipo", "num_cpu_cores", "num_cpu_threads", "num_cpu_sockets", "sistema",
    "versao_blender_major", "versao_blender_minor",
    "hw_fabricante", "hw_familia", "hw_ano", "hw_classe", "hw_notebook", "hw_nucleos_gpu", "hw_vram_gb",
) + DISPOSITIVO
HOST = ("host_familia", "host_ano", "host_classe") + HOSPEDEIRA

CENA_VAZAO = (
    "log_triangulos", "num_instancias", "max_niveis_subdiv", "cobertura_tela", "objetos_visiveis_camera",
    "profundidade_media", "num_objetos_curves", "num_objetos_volume", "usa_particulas", "usa_geometry_nodes",
    "cycles_max_bounces", "cycles_diffuse_bounces", "cycles_glossy_bounces", "cycles_transmission_bounces",
    "cycles_transparent_bounces", "cycles_volume_bounces", "cycles_volume_max_steps",
    "cycles_caustics_refl", "cycles_caustics_refr", "cycles_light_tree", "usa_motion_blur", "usa_dof",
    "num_materiais", "num_nodes_shader", "profundidade_max_shader", "num_mat_transparentes",
    "num_nodes_emissao", "num_shaders_caros", "num_nodes_volume",
    "num_luzes", "num_luz_point", "num_luz_sun", "num_luz_spot", "num_luz_area", "num_luzes_com_sombra",
    "area_emissora_total", "potencia_luzes_total", "usa_hdri", "mundo_com_volume", "num_nodes_mundo",
    "film_transparente",
    "cycles_adaptive", "cycles_adaptive_threshold", "cycles_min_samples",
)
CENA_CUSTO = (
    "num_objetos_mesh", "num_instancias", "log_triangulos", "num_modificadores", "max_niveis_subdiv",
    "usa_boolean", "usa_geometry_nodes", "usa_particulas", "num_objetos_volume",
    "num_texturas", "log_pixels_textura", "usa_hdri", "hdri_pixels",
    "num_materiais", "num_nodes_shader", "num_luzes",
    "cycles_denoise", "cycles_denoiser", "usa_compositor", "nodes_compositor", "num_view_layers",
    "pixels_totais",
)
COLUNAS_CENA = tuple(dict.fromkeys(CENA_VAZAO + CENA_CUSTO + ("amostras", "pixels_totais")))

ETAPAS = ("carregar", "preparar_finalizar")
VAZAO = CENA_VAZAO + MAQUINA
CUSTO = ("etapa",) + CENA_CUSTO + MAQUINA + HOST

CATEGORICAS = frozenset({"etapa", "dispositivo_tipo", "sistema", "hw_fabricante", "hw_familia",
                         "host_familia", "cycles_denoiser", "esp_arquitetura"})

ESQUEMAS = {"vazao": VAZAO, "custo_fixo": CUSTO}
COLUNA_ALVO = {"vazao": "log_vazao", "custo_fixo": "log_custo"}


def colunas (alvo: str) -> tuple[str, ...]:
    cols = ESQUEMAS[alvo]
    proibidas = PROIBIDAS_SEMPRE | (frozenset(NA_FORMULA) if alvo == "vazao" else frozenset())
    if invalidas := sorted(set(cols) & proibidas):
        raise ValueError(f"Colunas proibidas no esquema {alvo}: {invalidas}")
    return cols


def sem_redundancia (df: pd.DataFrame, cols: tuple[str, ...]) -> tuple[str, ...]:
    cena = [c for c in cols if c in set(CENA_VAZAO + CENA_CUSTO) and c not in CATEGORICAS]
    por_cena = df.groupby("cena_sha256")[cena].first().apply(_numerico)
    fora = set(por_cena.columns[por_cena.nunique() <= 1])
    postos = por_cena.rank()
    mantidos = []
    for c in cena:
        if c in fora:
            continue
        if any(abs(postos[c].corr(postos[m])) > 1 - 1e-9 for m in mantidos):
            fora.add(c)
        else:
            mantidos.append(c)
    return tuple(c for c in cols if c not in fora)


def montar_X (
    df: pd.DataFrame,
    cols: tuple[str, ...],
    categorias: Optional[dict[str, list]] = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
    if faltando := [c for c in cols if c not in df.columns]:
        raise ValueError(f"Colunas ausentes: {faltando}")
    X = df.loc[:, list(cols)].copy()
    categorias = dict(categorias or {})
    for c in cols:
        if c in CATEGORICAS:
            valores = X[c].astype("string")
            if c not in categorias:
                categorias[c] = sorted(valores.dropna().unique().tolist())
            valores = valores.where(valores.isin(categorias[c]))
            X[c] = pd.Categorical(valores, categories=categorias[c])
        else:
            X[c] = _numerico(X[c])
    return X, categorias


def _numerico (s: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s) or s.dtype == bool:
        s = s.map(lambda v: 1.0 if v in (True, "True") else 0.0 if v in (False, "False") else v)
    return pd.to_numeric(s, errors="coerce").astype(float)
