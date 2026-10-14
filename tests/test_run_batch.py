"""
Testes da selecao de binario do Blender por versao do arquivo .blend.

Rodam sem o Blender instalado: `run_batch` nao importa `bpy`.
"""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract.run_batch import escolher_binario, versao_do_blend


def _escrever_blend(caminho: Path, versao: str, comprimido: bool = False) -> Path:
    """Gera um cabecalho .blend minimo: b'BLENDER' + ponteiro + endian + versao."""
    conteudo = b"\x28\xb5\x2f\xfd" if comprimido else b"BLENDER-v" + versao.encode()
    caminho.write_bytes(conteudo + struct.pack("<I", 0))
    return caminho


@pytest.mark.parametrize(
    "digitos, esperado",
    [("405", (4, 5)), ("420", (4, 20)), ("500", (5, 0)), ("510", (5, 10))],
)
def test_versao_do_blend_le_cabecalho(tmp_path, digitos, esperado):
    arquivo = _escrever_blend(tmp_path / "c.blend", digitos)
    assert versao_do_blend(arquivo) == esperado


def test_versao_do_blend_retorna_none_para_comprimido(tmp_path):
    arquivo = _escrever_blend(tmp_path / "z.blend", "500", comprimido=True)
    assert versao_do_blend(arquivo) is None


BINARIOS = {"4.2": "/opt/b42", "4.5": "/opt/b45", "5.1": "/opt/b51"}


@pytest.mark.parametrize(
    "versao, esperado",
    [
        ((4, 0), "/opt/b42"),  # abre no menor binario >= a versao do arquivo
        ((4, 2), "/opt/b42"),
        ((4, 3), "/opt/b45"),
        ((5, 0), "/opt/b51"),
        ((5, 1), "/opt/b51"),
        ((6, 0), "/opt/b51"),  # mais novo que tudo: tenta o binario mais recente
    ],
)
def test_escolher_binario(versao, esperado):
    assert escolher_binario(versao, BINARIOS, "padrao") == esperado


def test_escolher_binario_sem_mapa_usa_padrao():
    assert escolher_binario((5, 0), {}, "padrao") == "padrao"
    assert escolher_binario(None, BINARIOS, "padrao") == "padrao"
