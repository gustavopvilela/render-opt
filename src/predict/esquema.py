"""
O que entra em cada modelo. Nenhum outro arquivo escolhe colunas.

Ter as listas num lugar só permite testar que nada proibido entra (vazamento)
e gravar a lista exata junto do modelo treinado.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd

# Nunca são features: identificadores (a cena entra pelos atributos), alvos, pós-render
CHAVES = ("origem", "fonte", "maquina", "cena", "cena_sha256", "arquivo_blend", "frame",
          "dispositivo_nome", "cpu_nome", "medicao_id", "trabalhador_id", "job_id", "lote_id", "peso")
ALVOS = ("vazao_amostras_pixel_s", "log_vazao", "custo_fixo_s", "log_custo_fixo", "log_custo",
         "carregar_s", "log_carregar", "preparar_finalizar_s", "log_preparar")
POS_RENDER = ("n_medicoes", "medido_num_amostras", "medido_time_for_samples", "medido_total_render_time",
              "medido_samples_per_minute", "medido_pico_memoria_mb", "tempo_total_s", "tempo_amostragem_s",
              "amostragem_s", "tempo_preparacao_s", "tempo_blender_s", "tempo_gravacao_s",
              "instante_ultima_amostra_s", "amostra_final", "amostras_total", "memoria_max_mb",
              "tempo_previsto_s")
# Já entram na fórmula p = c + S*W*H / v
NA_FORMULA = ("amostras", "amostras_x_pixels", "log_amostras_x_pixels", "res_x", "res_y", "pixels_totais")

# Auxiliares do OpenBenchmarking
AUXILIARES_OB = ("log_vazao_limite_inferior", "log_tempo_render", "tempo_render_s", "custo_fixo_imputado_s",
                 "carregamento_s", "carregamento_atipico", "vazao_origem", "custo_fixo_origem", "carregamento_origem")

PROIBIDAS_SEMPRE = frozenset(CHAVES + ALVOS + POS_RENDER + AUXILIARES_OB)

# Máquina
MAQUINA = (
    "dispositivo_tipo", "num_cpu_cores", "num_cpu_threads", "num_cpu_sockets", "sistema",
    "versao_blender_major", "versao_blender_minor",
    "hw_fabricante", "hw_familia", "hw_ano", "hw_classe", "hw_notebook", "hw_nucleos_gpu", "hw_vram_gb",
)
# CPU hospedeira
HOST = ("host_familia", "host_ano", "host_classe")

# Cena
CENA_VAZAO = (
    # geometria visível e custo de traçado
    "log_triangulos", "num_instancias", "max_niveis_subdiv", "cobertura_tela", "objetos_visiveis_camera",
    "profundidade_media", "num_objetos_curves", "num_objetos_volume", "usa_particulas", "usa_geometry_nodes",
    # transporte de luz
    "cycles_max_bounces", "cycles_diffuse_bounces", "cycles_glossy_bounces", "cycles_transmission_bounces",
    "cycles_transparent_bounces", "cycles_volume_bounces", "cycles_volume_max_steps",
    "cycles_caustics_refl", "cycles_caustics_refr", "cycles_light_tree", "usa_motion_blur", "usa_dof",
    # materiais, luzes e mundo
    "num_materiais", "num_nodes_shader", "profundidade_max_shader", "num_mat_transparentes",
    "num_nodes_emissao", "num_shaders_caros", "num_nodes_volume",
    "num_luzes", "num_luz_point", "num_luz_sun", "num_luz_spot", "num_luz_area", "num_luzes_com_sombra",
    "area_emissora_total", "potencia_luzes_total", "usa_hdri", "mundo_com_volume", "num_nodes_mundo",
    "film_transparente",
    # amostragem adaptativa
    "cycles_adaptive", "cycles_adaptive_threshold", "cycles_min_samples",
)
CENA_CUSTO = (
    # carregar e preparar: geometria, texturas, BVH
    "num_objetos_mesh", "num_instancias", "log_triangulos", "num_modificadores", "max_niveis_subdiv",
    "usa_boolean", "usa_geometry_nodes", "usa_particulas", "num_objetos_volume",
    "num_texturas", "log_pixels_textura", "usa_hdri", "hdri_pixels",
    "num_materiais", "num_nodes_shader", "num_luzes",
    # depois da amostragem: denoise, composição, gravação (crescem com a resolução)
    "cycles_denoise", "cycles_denoiser", "usa_compositor", "nodes_compositor", "num_view_layers",
    "pixels_totais",
)
# Colunas de cena guardadas no dataset
COLUNAS_CENA = tuple(dict.fromkeys(CENA_VAZAO + CENA_CUSTO + ("amostras", "pixels_totais")))

# Modelos
ETAPAS = ("carregar", "preparar_finalizar")
VAZAO = CENA_VAZAO + MAQUINA
CUSTO = ("etapa",) + CENA_CUSTO + MAQUINA + HOST

CATEGORICAS = frozenset({"etapa", "dispositivo_tipo", "sistema", "hw_fabricante", "hw_familia",
                         "host_familia", "cycles_denoiser"})

ESQUEMAS = {"vazao": VAZAO, "custo_fixo": CUSTO}
COLUNA_ALVO = {"vazao": "log_vazao", "custo_fixo": "log_custo"}


def colunas (alvo: str) -> tuple[str, ...]:
    cols = ESQUEMAS[alvo]
    proibidas = PROIBIDAS_SEMPRE | (frozenset(NA_FORMULA) if alvo == "vazao" else frozenset())
    if invalidas := sorted(set(cols) & proibidas):
        raise ValueError(f"Colunas proibidas no esquema {alvo}: {invalidas}")
    return cols


def montar_X (
    df: pd.DataFrame,
    cols: tuple[str, ...],
    categorias: Optional[dict[str, list]] = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """
    Seleciona as colunas na ordem declarada e acerta os tipos.

    - booleanas viram 0/1 (float, para aceitar NaN);
    - categóricas viram pd.Categorical com as categorias do TREINO. Na inferência,
      passe as categorias gravadas no modelo: uma categoria nunca vista vira NaN
      em vez de mudar a codificação das outras.
    :return: (X, categorias usadas)
    """
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
    """Booleanas (inclusive "True"/"False" lidas de CSV) viram 1/0; o resto, float."""
    if not pd.api.types.is_numeric_dtype(s) or s.dtype == bool:
        s = s.map(lambda v: 1.0 if v in (True, "True") else 0.0 if v in (False, "False") else v)
    return pd.to_numeric(s, errors="coerce").astype(float)
