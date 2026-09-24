"""
Coleta de atributos de cenas Blender para o modelo preditivo de tempo de
renderizacao.

A saida e uma linha achatada (`dict[str, int|float|str|bool]`) por frame, que e
o formato consumido diretamente por pandas/scikit-learn. Detalhes por objeto e
por material sao agregados aqui e nao exportados individualmente: o modelo
precisa de features de tamanho fixo, nao de um dump da cena.
"""

from __future__ import annotations

import math
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from . import compat

# Shaders cujo custo por amostra e desproporcional no path tracing.
SHADERS_CAROS = {
    "BSDF_GLASS",
    "BSDF_REFRACTION",
    "BSDF_TRANSPARENT",
    "SUBSURFACE_SCATTERING",
    "PRINCIPLED_VOLUME",
    "VOLUME_SCATTER",
    "VOLUME_ABSORPTION",
    "BSDF_HAIR",
    "BSDF_HAIR_PRINCIPLED",
}

GRADE_COBERTURA = 32  # resolucao da grade usada no proxy de cobertura de tela

# Teto de subdivisao ao avaliar a malha. Nivel 6 em uma cena grande estoura a
# memoria da maquina de extracao; a linha marca `subdiv_truncado` quando o teto
# e aplicado, para que esses casos possam ser filtrados na analise.
MAX_SUBDIV_AVALIADO = 4


def _render_dimensions(scene: bpy.types.Scene) -> tuple[int, int, int]:
    render = scene.render
    escala = render.resolution_percentage / 100.0
    largura = int(render.resolution_x * escala)
    altura = int(render.resolution_y * escala)
    return largura, altura, largura * altura


def coletar_config_render(scene: bpy.types.Scene) -> dict[str, Any]:
    """Parametros do motor: o bloco de features mais preditivo do conjunto."""
    render = scene.render
    largura, altura, pixels = _render_dimensions(scene)
    familia = compat.engine_family(render.engine)
    amostras = compat.render_samples(scene)

    linha: dict[str, Any] = {
        "motor": familia,
        "motor_id": render.engine,
        "res_x": largura,
        "res_y": altura,
        "pixels_totais": pixels,
        "amostras": amostras,
        # Feature dominante: o custo do path tracing e ~linear em raios lancados.
        "amostras_x_pixels": float(amostras) * pixels,
        "log_amostras_x_pixels": math.log1p(float(amostras) * pixels),
        "usa_motion_blur": bool(render.use_motion_blur),
        "usa_sombras": compat.shadows_enabled(scene),
        "film_transparente": bool(compat.prop(render, "film_transparent", False)),
        "threads_render": int(compat.prop(render, "threads", 0)),
        "usa_persistent_data": bool(compat.prop(render, "use_persistent_data", False)),
        "num_view_layers": len(scene.view_layers),
        "usa_compositor": False,
        "nodes_compositor": 0,
        # `use_sequencer` vem ligado por padrao mesmo sem strips; so conta se o
        # editor de video realmente tiver conteudo a compor.
        "usa_sequencer": bool(compat.prop(render, "use_sequencer", False))
        and bool(getattr(getattr(scene, "sequence_editor", None), "strips", None)),
    }

    arvore_comp = compat.compositor_node_tree(scene)
    if arvore_comp is not None:
        linha["usa_compositor"] = True
        linha["nodes_compositor"] = len(arvore_comp.nodes)

    # Cycles
    cycles = compat.cycles_settings(scene)
    linha.update(
        {
            "cycles_device": str(compat.prop(cycles, "device", "N/A")),
            "cycles_max_bounces": int(compat.prop(cycles, "max_bounces", 0)),
            "cycles_diffuse_bounces": int(compat.prop(cycles, "diffuse_bounces", 0)),
            "cycles_glossy_bounces": int(compat.prop(cycles, "glossy_bounces", 0)),
            "cycles_transmission_bounces": int(
                compat.prop(cycles, "transmission_bounces", 0)
            ),
            "cycles_transparent_bounces": int(
                compat.prop(cycles, "transparent_max_bounces", 0)
            ),
            "cycles_volume_bounces": int(compat.prop(cycles, "volume_bounces", 0)),
            "cycles_volume_max_steps": int(compat.prop(cycles, "volume_max_steps", 0)),
            "cycles_adaptive": bool(compat.prop(cycles, "use_adaptive_sampling", False)),
            "cycles_adaptive_threshold": float(
                compat.prop(cycles, "adaptive_threshold", 0.0)
            ),
            "cycles_min_samples": int(compat.prop(cycles, "adaptive_min_samples", 0)),
            "cycles_time_limit": float(compat.prop(cycles, "time_limit", 0.0)),
            "cycles_denoise": bool(compat.prop(cycles, "use_denoising", False)),
            "cycles_denoiser": str(compat.prop(cycles, "denoiser", "N/A")),
            "cycles_light_tree": bool(compat.prop(cycles, "use_light_tree", False)),
            "cycles_caustics_refl": bool(
                compat.prop(cycles, "caustics_reflective", False)
            ),
            "cycles_caustics_refr": bool(
                compat.prop(cycles, "caustics_refractive", False)
            ),
        }
    )

    # EEVEE
    eevee = compat.eevee_settings(scene)
    linha.update(
        {
            "eevee_raytracing": bool(compat.prop(eevee, "use_raytracing", False)),
            "eevee_volumetric_shadows": bool(
                compat.prop(eevee, "use_volumetric_shadows", False)
            ),
            "eevee_shadow_ray_count": int(compat.prop(eevee, "shadow_ray_count", 0)),
            "eevee_shadow_step_count": int(compat.prop(eevee, "shadow_step_count", 0)),
        }
    )
    return linha


@contextmanager
def qualidade_de_render(scene: bpy.types.Scene) -> Iterator[bool]:
    """
    Alinha temporariamente o estado de viewport ao estado de render.

    O depsgraph acessivel por script e o de VIEWPORT: um Subsurf com
    `render_levels = 3` e `levels = 1` seria contado com 1/16 dos poligonos
    reais, e objetos com `hide_viewport` marcado sumiriam da contagem mesmo
    aparecendo no render. Sem este ajuste as features de geometria descrevem
    uma cena diferente daquela que sera cronometrada no benchmark.

    Todas as alteracoes sao revertidas na saida e o arquivo nunca e salvo.
    """
    alteracoes: list[tuple[Any, str, Any]] = []
    truncou = False

    def guardar(alvo: Any, atributo: str, valor: Any) -> None:
        alteracoes.append((alvo, atributo, getattr(alvo, atributo)))
        setattr(alvo, atributo, valor)

    for obj in scene.objects:
        if obj.hide_render:
            continue
        if obj.hide_viewport:
            guardar(obj, "hide_viewport", False)
        for modificador in getattr(obj, "modifiers", []) or []:
            if not modificador.show_render:
                if modificador.show_viewport:
                    guardar(modificador, "show_viewport", False)
                continue
            if not modificador.show_viewport:
                guardar(modificador, "show_viewport", True)
            if modificador.type in {"SUBSURF", "MULTIRES"}:
                alvo = min(int(modificador.render_levels), MAX_SUBDIV_AVALIADO)
                truncou = truncou or alvo < int(modificador.render_levels)
                if int(modificador.levels) != alvo:
                    guardar(modificador, "levels", alvo)

    for colecao in bpy.data.collections:
        if not colecao.hide_render and colecao.hide_viewport:
            guardar(colecao, "hide_viewport", False)

    try:
        yield truncou
    finally:
        for alvo, atributo, valor in reversed(alteracoes):
            try:
                setattr(alvo, atributo, valor)
            except (AttributeError, ReferenceError):
                pass  # datablock removido durante a avaliacao


def coletar_geometria(depsgraph: bpy.types.Depsgraph) -> dict[str, Any]:
    """
    Contabiliza a geometria *avaliada*: pos-modificadores e com instancias
    expandidas. E o unico numero que corresponde ao que o renderizador
    realmente processa -- contar `len(obj.data.vertices)` subestima em ordens
    de magnitude qualquer cena com Subdivision Surface ou particulas.
    """
    total_vertices = 0
    total_poligonos = 0
    total_triangulos = 0
    objetos_mesh = 0
    instancias = 0
    max_subdiv = 0
    total_modificadores = 0
    tipos_modificadores: set[str] = set()
    objetos_curve = 0
    objetos_volume = 0

    for inst in depsgraph.object_instances:
        obj = inst.object
        if obj is None:
            continue
        if obj.type == "CURVES":
            objetos_curve += 1
        if obj.type == "VOLUME":
            objetos_volume += 1
        if obj.type != "MESH":
            continue

        malha = obj.data
        if malha is None:
            continue

        if inst.is_instance:
            instancias += 1
        else:
            objetos_mesh += 1
            for modificador in getattr(obj.original, "modifiers", []) or []:
                if not modificador.show_render:
                    continue
                total_modificadores += 1
                tipos_modificadores.add(modificador.type)
                if modificador.type == "SUBSURF":
                    max_subdiv = max(max_subdiv, int(modificador.render_levels))

        n_poligonos = len(malha.polygons)
        total_vertices += len(malha.vertices)
        total_poligonos += n_poligonos
        # Identidade: cada poligono de n vertices vira n-2 triangulos, e a soma
        # de n sobre todos os poligonos e exatamente len(loops). Equivale a
        # `sum(len(p.vertices) - 2 for p in malha.polygons)` mas sem laco Python
        # sobre RNA -- verificado identico nas 3 cenas do benchmark, e ~10^4x
        # mais rapido no junkshop (9,3 s -> 0,0 s por frame).
        total_triangulos += len(malha.loops) - 2 * n_poligonos

    return {
        "num_objetos_mesh": objetos_mesh,
        "num_instancias": instancias,
        "vertices_avaliados": total_vertices,
        "poligonos_avaliados": total_poligonos,
        "triangulos_avaliados": total_triangulos,
        "log_triangulos": math.log1p(total_triangulos),
        "num_modificadores": total_modificadores,
        "max_niveis_subdiv": max_subdiv,
        "usa_boolean": "BOOLEAN" in tipos_modificadores,
        "usa_particulas": "PARTICLE_SYSTEM" in tipos_modificadores,
        "usa_geometry_nodes": "NODES" in tipos_modificadores,
        "num_objetos_curves": objetos_curve,
        "num_objetos_volume": objetos_volume,
    }


def coletar_iluminacao(scene: bpy.types.Scene) -> dict[str, Any]:
    """Contagem de luzes por tipo e area emissora total."""
    contagem = {"POINT": 0, "SUN": 0, "SPOT": 0, "AREA": 0}
    area_emissora = 0.0
    potencia_total = 0.0
    sombras_ativas = 0

    for obj in scene.objects:
        if obj.type != "LIGHT" or obj.hide_render:
            continue
        luz = obj.data
        contagem[luz.type] = contagem.get(luz.type, 0) + 1
        potencia_total += float(compat.prop(luz, "energy", 0.0))
        if compat.prop(luz, "use_shadow", True):
            sombras_ativas += 1
        if luz.type == "AREA":
            area_emissora += float(luz.size) * float(compat.prop(luz, "size_y", luz.size))

    total = sum(contagem.values())
    return {
        "num_luzes": total,
        "num_luz_point": contagem["POINT"],
        "num_luz_sun": contagem["SUN"],
        "num_luz_spot": contagem["SPOT"],
        "num_luz_area": contagem["AREA"],
        "num_luzes_com_sombra": sombras_ativas,
        "area_emissora_total": round(area_emissora, 4),
        "potencia_luzes_total": round(potencia_total, 4),
    }


def coletar_materiais(scene: bpy.types.Scene) -> dict[str, Any]:
    """
    Percorre apenas os materiais efetivamente usados pela cena. O extrator
    antigo varria `bpy.data.materials` inteiro, contando materiais orfaos e de
    outras cenas do mesmo arquivo.
    """
    materiais: set[bpy.types.Material] = set()
    for obj in scene.objects:
        if obj.hide_render:
            continue
        for slot in obj.material_slots:
            if slot.material is not None:
                materiais.add(slot.material)

    total_nodes = 0
    profundidade_max = 0
    num_transparentes = 0
    num_emissivos = 0
    num_shaders_caros = 0
    texturas: set[str] = set()
    pixels_textura = 0
    num_volumes = 0

    for material in materiais:
        arvore = getattr(material, "node_tree", None)
        if compat.material_is_transparent(material):
            num_transparentes += 1
        if arvore is None:
            continue

        total_nodes += len(arvore.nodes)
        profundidade_max = max(profundidade_max, _profundidade_arvore(arvore))

        for node in arvore.nodes:
            if node.type in SHADERS_CAROS:
                num_shaders_caros += 1
            if node.type in {"PRINCIPLED_VOLUME", "VOLUME_SCATTER", "VOLUME_ABSORPTION"}:
                num_volumes += 1
            if node.type == "EMISSION":
                num_emissivos += 1
            if node.type == "TEX_IMAGE" and node.image is not None:
                if node.image.name not in texturas:
                    texturas.add(node.image.name)
                    largura, altura = compat.image_resolution(node.image)
                    pixels_textura += largura * altura

    return {
        "num_materiais": len(materiais),
        "num_nodes_shader": total_nodes,
        "profundidade_max_shader": profundidade_max,
        "num_mat_transparentes": num_transparentes,
        "num_nodes_emissao": num_emissivos,
        "num_shaders_caros": num_shaders_caros,
        "num_nodes_volume": num_volumes,
        "num_texturas": len(texturas),
        "pixels_textura_total": pixels_textura,
        "log_pixels_textura": math.log1p(pixels_textura),
    }


def _profundidade_arvore(arvore: Any) -> int:
    """Maior distancia em nos ate a saida do material (proxy de complexidade)."""
    saidas = [n for n in arvore.nodes if n.type == "OUTPUT_MATERIAL"]
    if not saidas:
        return 0

    profundidade = 0
    visitados: set[str] = set()
    fronteira = [(n, 0) for n in saidas]
    while fronteira:
        node, nivel = fronteira.pop()
        if node.name in visitados or nivel > 64:  # corta ciclos e node groups fundos
            continue
        visitados.add(node.name)
        profundidade = max(profundidade, nivel)
        for entrada in node.inputs:
            for link in entrada.links:
                fronteira.append((link.from_node, nivel + 1))
    return profundidade


def coletar_mundo(scene: bpy.types.Scene) -> dict[str, Any]:
    """Ambiente global: HDRI e nevoa volumetrica dominam o custo em Cycles."""
    linha: dict[str, Any] = {
        "usa_hdri": False,
        "hdri_pixels": 0,
        "mundo_com_volume": False,
        "num_nodes_mundo": 0,
    }
    mundo = scene.world
    arvore = getattr(mundo, "node_tree", None) if mundo else None
    if arvore is None:
        return linha

    linha["num_nodes_mundo"] = len(arvore.nodes)
    for node in arvore.nodes:
        if node.type == "TEX_ENVIRONMENT" and node.image is not None:
            largura, altura = compat.image_resolution(node.image)
            linha["usa_hdri"] = True
            linha["hdri_pixels"] = max(linha["hdri_pixels"], largura * altura)
        if node.type in {"VOLUME_SCATTER", "VOLUME_ABSORPTION", "PRINCIPLED_VOLUME"}:
            linha["mundo_com_volume"] = True
    return linha


def coletar_camera(scene: bpy.types.Scene) -> dict[str, Any]:
    camera = scene.camera
    if camera is None or camera.type != "CAMERA":
        return {
            "tem_camera": False,
            "distancia_focal": 0.0,
            "usa_dof": False,
            "tipo_lente": "N/A",
        }
    dados = camera.data
    return {
        "tem_camera": True,
        "distancia_focal": float(compat.prop(dados, "lens", 0.0)),
        "usa_dof": bool(compat.prop(getattr(dados, "dof", None), "use_dof", False)),
        "tipo_lente": str(compat.prop(dados, "type", "PERSP")),
    }


def coletar_cobertura_tela(
    scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph
) -> dict[str, Any]:
    """
    Proxy analitico de "quanto da tela tem geometria".

    Projeta a caixa delimitadora de cada instancia avaliada no espaco da camera
    e marca as celulas de uma grade 32x32. Distingue "objeto pequeno ao longe"
    de "cena preenchida" -- distincao que a contagem de poligonos nao captura e
    que explica boa parte da variancia do tempo entre frames de uma mesma
    animacao. Custa milissegundos, ao contrario de um render de preview.
    """
    vazio = {
        "cobertura_tela": 0.0,
        "objetos_visiveis_camera": 0,
        "profundidade_media": 0.0,
    }
    camera = scene.camera
    if camera is None or camera.type != "CAMERA":
        return vazio

    grade = [[False] * GRADE_COBERTURA for _ in range(GRADE_COBERTURA)]
    visiveis = 0
    soma_profundidade = 0.0

    for inst in depsgraph.object_instances:
        obj = inst.object
        if obj is None or obj.type not in {"MESH", "CURVES", "VOLUME", "SURFACE"}:
            continue
        bound_box = getattr(obj, "bound_box", None)
        if bound_box is None:
            continue

        matriz = inst.matrix_world
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for canto in bound_box:
            ndc = world_to_camera_view(scene, camera, matriz @ Vector(canto))
            xs.append(ndc.x)
            ys.append(ndc.y)
            zs.append(ndc.z)

        if max(zs) <= 0.0:  # inteiramente atras da camera
            continue
        x0, x1 = max(min(xs), 0.0), min(max(xs), 1.0)
        y0, y1 = max(min(ys), 0.0), min(max(ys), 1.0)
        if x1 <= x0 or y1 <= y0:
            continue

        visiveis += 1
        soma_profundidade += max(min(zs), 0.0)
        for cx in range(int(x0 * GRADE_COBERTURA), min(int(x1 * GRADE_COBERTURA) + 1, GRADE_COBERTURA)):
            for cy in range(int(y0 * GRADE_COBERTURA), min(int(y1 * GRADE_COBERTURA) + 1, GRADE_COBERTURA)):
                grade[cx][cy] = True

    ocupadas = sum(1 for coluna in grade for celula in coluna if celula)
    return {
        "cobertura_tela": round(ocupadas / (GRADE_COBERTURA**2), 4),
        "objetos_visiveis_camera": visiveis,
        "profundidade_media": round(soma_profundidade / visiveis, 4) if visiveis else 0.0,
    }


def coletar_animacao(scene: bpy.types.Scene) -> dict[str, Any]:
    """Densidade de animacao: proxy de quanto a cena muda entre frames."""
    total_keyframes = 0
    total_fcurves = 0
    for action in bpy.data.actions:
        for fcurve in compat.action_fcurves(action):
            total_fcurves += 1
            total_keyframes += len(fcurve.keyframe_points)

    return {
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "total_frames": max(scene.frame_end - scene.frame_start + 1, 1),
        "num_acoes": len(bpy.data.actions),
        "num_fcurves": total_fcurves,
        "num_keyframes": total_keyframes,
        "usa_simulacao_fisica": any(
            m.type in {"FLUID", "CLOTH", "SOFT_BODY", "COLLISION", "DYNAMIC_PAINT"}
            for obj in scene.objects
            for m in (getattr(obj, "modifiers", []) or [])
        ),
    }


def coletar_invariantes(scene: bpy.types.Scene) -> dict[str, Any]:
    """
    Features que NAO mudam ao longo da timeline: configuracao de render,
    materiais, texturas, mundo e estatisticas de animacao.

    Sao estruturais -- descrevem o arquivo, nao o instante. Calcula-las uma vez
    por cena em vez de uma vez por frame e o que torna a extracao viavel em
    cenas pesadas: no junkshop `coletar_materiais` sozinho custa ~11 s, o que
    daria 45 min so nessa etapa ao longo de 250 frames.
    """
    invariantes: dict[str, Any] = {}
    invariantes.update(coletar_config_render(scene))
    invariantes.update(coletar_materiais(scene))
    invariantes.update(coletar_mundo(scene))
    invariantes.update(coletar_animacao(scene))
    return invariantes


def extrair_frame(
    scene: bpy.types.Scene,
    frame: int,
    invariantes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Extrai a linha de features de um frame especifico.

    `frame_set` reavalia o depsgraph, entao geometria animada, instancias e
    cobertura de tela sao medidas no estado real daquele frame -- e por isso
    que ha uma linha por frame, e nao uma por arquivo.

    `invariantes` permite reaproveitar as features estruturais entre frames.
    Se omitido, sao recalculadas -- correto, porem lento.
    """
    scene.frame_set(frame)

    linha: dict[str, Any] = {
        "arquivo_blend": bpy.data.filepath or "nao_salvo",
        "cena": scene.name,
        "frame": frame,
        "versao_blender": bpy.app.version_string,
        "versao_blender_major": compat.VERSION[0],
        "versao_arquivo": ".".join(str(v) for v in bpy.data.version),
    }
    with qualidade_de_render(scene) as truncou:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        linha.update(coletar_geometria(depsgraph))
        linha.update(coletar_cobertura_tela(scene, depsgraph))
    linha["subdiv_truncado"] = truncou

    linha.update(invariantes if invariantes is not None else coletar_invariantes(scene))
    # Iluminacao e camera ficam por frame de proposito: potencia de luz,
    # distancia focal e profundidade de campo sao rotineiramente animadas.
    linha.update(coletar_iluminacao(scene))
    linha.update(coletar_camera(scene))
    return linha


def extrair_cena(
    scene: bpy.types.Scene,
    frames: Optional[list[int]] = None,
    passo: int = 1,
) -> list[dict[str, Any]]:
    """Extrai uma linha por frame solicitado (padrao: a timeline inteira)."""
    if frames is None:
        frames = list(range(scene.frame_start, scene.frame_end + 1, max(passo, 1)))

    invariantes = coletar_invariantes(scene)

    linhas: list[dict[str, Any]] = []
    inicio = time.monotonic()
    for i, frame in enumerate(frames, start=1):
        linhas.append(extrair_frame(scene, frame, invariantes))
        # Progresso em stderr: sem isto uma cena de 250 frames fica minutos
        # muda e parece travada.
        decorrido = time.monotonic() - inicio
        restante = decorrido / i * (len(frames) - i)
        print(
            f"[RENDEROPT] {scene.name}: frame {frame} "
            f"({i}/{len(frames)}, {decorrido:.0f}s decorridos, "
            f"~{restante:.0f}s restantes)",
            file=sys.stderr,
            flush=True,
        )
    return linhas
