"""
Camada de compatibilidade da API do Blender entre as series 4.x e 5.x.

Todo acesso a propriedade que mudou de nome, mudou de lugar ou foi removida
entre as versoes deve passar por este modulo. O restante do extrator nao deve
conter nenhum `hasattr`/`getattr` defensivo espalhado.

Diferencas tratadas (verificadas em Blender 5.1.0):

- Motor EEVEE: id `BLENDER_EEVEE` (<=4.1 legado e >=5.0 novo) e
  `BLENDER_EEVEE_NEXT` (4.2 ate 4.5).
- `Scene.node_tree` (compositor) foi removido em 5.x, substituido por
  `Scene.compositing_node_group`; `Scene.use_nodes` esta depreciado.
- `Action.fcurves` foi removido em 5.x. A partir de 4.4 as acoes sao
  "slotted": `Action.layers -> Strip.channelbag(slot) -> fcurves`.
- `RenderSettings.use_shadows` nao existe em nenhuma versao; sombras sao
  configuradas por motor (`Scene.eevee.use_shadows`).
- `CyclesRenderSettings.debug_bvh_method` foi removido nas versoes recentes.
- `Material.blend_method` convive com `Material.surface_render_method` (4.2+),
  que e a propriedade correta no EEVEE Next/5.x.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import bpy

VERSION: tuple[int, int, int] = tuple(bpy.app.version)  # type: ignore[assignment]

ENGINE_CYCLES = "CYCLES"
ENGINE_WORKBENCH = "BLENDER_WORKBENCH"
EEVEE_ENGINES = frozenset({"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"})


def is_eevee(engine: str) -> bool:
    return engine in EEVEE_ENGINES


def engine_family(engine: str) -> str:
    """Normaliza o id do motor para uma categoria estavel entre versoes."""
    if engine == ENGINE_CYCLES:
        return "CYCLES"
    if is_eevee(engine):
        return "EEVEE"
    if engine == ENGINE_WORKBENCH:
        return "WORKBENCH"
    return "OUTRO"


def prop(owner: Any, name: str, default: Any = None) -> Any:
    """Le uma propriedade opcional sem quebrar em versoes que nao a possuem."""
    if owner is None:
        return default
    value = getattr(owner, name, None)
    return default if value is None else value


def cycles_settings(scene: bpy.types.Scene) -> Optional[Any]:
    """`scene.cycles` so existe com o add-on Cycles habilitado."""
    return getattr(scene, "cycles", None)


def eevee_settings(scene: bpy.types.Scene) -> Optional[Any]:
    return getattr(scene, "eevee", None)


def render_samples(scene: bpy.types.Scene) -> int:
    """Numero de amostras efetivo do motor ativo (1 quando nao se aplica)."""
    family = engine_family(scene.render.engine)
    if family == "CYCLES":
        return int(prop(cycles_settings(scene), "samples", 1))
    if family == "EEVEE":
        return int(prop(eevee_settings(scene), "taa_render_samples", 64))
    return 1


def shadows_enabled(scene: bpy.types.Scene) -> bool:
    """
    `RenderSettings.use_shadows` nunca existiu: o extrator antigo sempre
    retornava True. No Cycles as sombras sao intrinsecas ao path tracing; no
    EEVEE a chave e `scene.eevee.use_shadows` (4.2+).
    """
    family = engine_family(scene.render.engine)
    if family == "CYCLES":
        return True
    if family == "EEVEE":
        return bool(prop(eevee_settings(scene), "use_shadows", True))
    return False


def compositor_node_tree(scene: bpy.types.Scene) -> Optional[Any]:
    """Arvore do compositor, cujo caminho mudou em 5.x."""
    group = getattr(scene, "compositing_node_group", None)
    if group is not None:
        return group
    if getattr(scene, "use_nodes", False):
        return getattr(scene, "node_tree", None)
    return None


def material_is_transparent(material: bpy.types.Material) -> bool:
    """
    Em 4.2+ o EEVEE Next usa `surface_render_method` ('DITHERED'/'BLENDED');
    `blend_method` permanece por compatibilidade nas versoes anteriores.
    """
    method = getattr(material, "surface_render_method", None)
    if method is not None:
        return str(method) == "BLENDED"
    return str(getattr(material, "blend_method", "OPAQUE")) != "OPAQUE"


def action_fcurves(action: bpy.types.Action) -> Iterable[Any]:
    """
    Itera as f-curves de uma acao nos dois modelos de dados:
    legado (`action.fcurves`, <=4.3) e slotted (`layers`, >=4.4).
    """
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy
        return

    slots = list(getattr(action, "slots", []) or [])
    for layer in getattr(action, "layers", []) or []:
        for strip in getattr(layer, "strips", []) or []:
            bags = getattr(strip, "channelbags", None)
            if bags is not None:
                for bag in bags:
                    yield from bag.fcurves
                continue
            getter = getattr(strip, "channelbag", None)
            if getter is None:
                continue
            for slot in slots:
                bag = getter(slot, ensure=False)
                if bag is not None:
                    yield from bag.fcurves


def cycles_device_preferences() -> Optional[Any]:
    """Preferencias do add-on Cycles, ausentes se o add-on estiver desligado."""
    addons = bpy.context.preferences.addons
    entry = addons.get("cycles")
    return getattr(entry, "preferences", None) if entry is not None else None


def image_resolution(image: bpy.types.Image) -> tuple[int, int]:
    """
    `image.size` retorna (0, 0) quando o buffer ainda nao foi carregado
    (comum em execucao headless com texturas linkadas).
    """
    width, height = int(image.size[0]), int(image.size[1])
    if width and height:
        return width, height
    try:
        image.reload()
    except RuntimeError:
        return 0, 0
    return int(image.size[0]), int(image.size[1])
