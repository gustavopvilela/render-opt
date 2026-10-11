"""
Imita o Blender na linha de comando, para testar o trabalhador sem Blender.
"""
import os
import sys
import time
import zlib
import struct

def png_1x1() -> bytes:
    def bloco(tipo: bytes, dados: bytes) -> bytes:
        return struct.pack(">I", len(dados)) + tipo + dados + struct.pack(">I", zlib.crc32(tipo + dados))
    return (b"\x89PNG\r\n\x1a\n" + bloco(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + bloco(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + bloco(b"IEND", b""))


def lista(nome: str) -> set[int]:
    return {int(x) for x in os.environ.get(nome, "").split(",") if x}


args = sys.argv[1:]
saida = args[args.index("-o") + 1]
frame = int(args[args.index("-f") + 1])
preparo = float(os.environ.get("FALSO_PREPARO_S", 0.2))
por_amostra = float(os.environ.get("FALSO_AMOSTRA_S", 0.05))
n = int(os.environ.get("FALSO_AMOSTRAS", 10))
if frame in lista("FALSO_LENTOS"):
    por_amostra *= 20

print("Blender 5.1.0 (falso)", flush=True)
time.sleep(preparo)
for i in range(n + 1):
    print(f"00:00.000  render | Fra: {frame} | Mem: {100 + i}M | Sample {i}/{n}", flush=True)
    if frame in lista("FALSO_TRAVAR") and i == n // 2:
        time.sleep(3600)
    if i < n:
        time.sleep(por_amostra)

if frame in lista("FALSO_FALHAR"):
    print("00:00.000  reports | ERROR falha simulada", flush=True)
    sys.exit(1)

caminho = saida.replace("######", f"{frame:06d}") + ".png"
with open(caminho, "wb") as arquivo:
    arquivo.write(png_1x1())
print(f"00:00.000  render | Saved: '{caminho}'", flush=True)
print("00:00.000  render | Time: 00:00.70 (Saving: 00:00.01)", flush=True)