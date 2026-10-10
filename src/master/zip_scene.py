"""
Roda dentro do Blender. Grava uma cópia autossuficiente
do .blend aberto. Comando:

blender --factory=startup -noaudio -b original.blend -P zip_scene.py --destino.blend

Autossuficiente: texturas e bibliotexas associadas vão
para dentro do arquivo. Sem isso, a cena aponta para um
caminho que não existe no trabalhador ao procurar.
"""

import sys
import bpy

destino = sys.argv[sys.argv.index("--") + 1]
bpy.ops.file.make_paths_absolute()

try:
    bpy.ops.object.make_local(type="ALL")
except RuntimeError as erro:
    print("RENDEROPT_AVISO make_local:", erro)

# Embute texturas, fontes, sons, etc.
bpy.ops.file.pack_all()

faltando = sorted({
    img.filepath for img in bpy.data.images
    if img.source in {"FILE", "SEQUENCE", "TILED"} and img.packed_file is None
})
ainda_ligados = sorted({lib.filepath for lib in bpy.data.libraries})
for caminho in faltando:
    print("RENDEROPT_FALTANDO", caminho)
for caminho in ainda_ligados:
    print("RENDEROPT_BIBLIOTECA", caminho)

bpy.ops.wm.save_as_mainfile(filepath=destino, copy=True, compress=True)
versao = bpy.app.version
print(f"RENDEROPT_VERSAO {versao[0]}.{versao[1]}")