# render-opt

Escalonamento preditivo de frames 3D em render farm heterogênea — TCC, Ciência da
Computação, IFMG Formiga.

## Estrutura

```
data/         .blend brutos (raw/) e dataset consolidado (processed/)
docs/         documentação de referência
experiments/  notebooks, resultados e figuras
src/
  extract/    extração de features de cenas Blender  [implementado]
  bench/      medição do tempo real de render por (frame, máquina)
  predict/    modelo de regressão do tempo de render
  schedule/   baselines + metaheurística para R||Cmax
  master/     nó mestre (distribuição de tarefas)
  worker/     nó trabalhador (executa o render)
tests/        testes que rodam sem o Blender instalado
```

## Instalação

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Extração de features

O extrator roda **dentro** do Blender (precisa de `bpy`), mas o orquestrador roda
no venv do projeto. Ele apenas lê a cena: não altera configurações nem renderiza.

Um arquivo:

```bash
blender -b cena.blend --factory-startup -noaudio \
  -P src/extract/extract_scene.py -- --saida data/raw/cena.jsonl
```

Um diretório inteiro, consolidando em CSV/Parquet:

```bash
python src/extract/run_batch.py data/raw/cenas \
  --saida data/processed/features.parquet --passo 5
```

### Arquivos 4.x e 5.x

`run_batch` lê a versão gravada no cabeçalho do `.blend` e escolhe o binário do
Blender compatível. Um `.blend` 5.x **não** abre em um Blender 4.x, então informe
um binário por série:

```bash
python src/extract/run_batch.py data/raw/cenas \
  --blender-versao 4.2=/opt/blender-4.2/blender \
  --blender-versao 4.5=/opt/blender-4.5/blender \
  --blender-versao 5.1="/c/Program Files/Blender Foundation/Blender 5.1/blender.exe"
```

Sem o mapa, usa `--blender` ou `$BLENDER_BIN`. Arquivos comprimidos (Zstd) não
expõem a versão no cabeçalho e caem no binário padrão — se falharem ao abrir,
a linha de erro aparece no resumo final do lote.

As diferenças de API entre as séries ficam concentradas em `src/extract/compat.py`;
o resto do extrator não faz checagem de versão.

## Documentação

| Arquivo | Conteúdo |
|---|---|
| `docs/especificacao-de-requisitos.md` | Introdução, tecnologias, requisitos funcionais/não funcionais, regras de negócio e casos de uso |
| `docs/obtencao-de-dados.md` | Como conseguir os `.blend` e os tempos medidos: fontes públicas com licença, o dataset CC0 do Blender Open Data, geração sintética, ferramentas de benchmark e baselines de referência |
| `docs/alvo-do-modelo.md` | O que o modelo deve prever: por que vazão e não segundos, o modelo de custo de três termos, como isso vira a matriz de tempos do escalonador e como treinar/avaliar |
| `docs/extracao-explicada.md` | Como o extrator funciona e como executá-lo |
| `docs/atributos-extraidos.md` | Dicionário das 94 colunas geradas |
| `docs/estimativa-do-blender.md` | A estimativa nativa do Blender: como capturá-la, por que não substitui o modelo, como usá-la como baseline |
| `docs/renderizacao-no-blender.md` | Como o Blender grava a saída: sequência de imagens, EXR multicamada, vídeo (FFmpeg) e áudio — e por que só o modo imagem é compatível com o escalonamento distribuído |
| `docs/tecnicas-ia-e-metaheuristicas.md` | Técnicas de IA e metaheurísticas candidatas, com referências, vantagens e desvantagens contextualizadas |

## Testes

```powershell
python -m pytest tests -q
```

Os testes não exigem Blender instalado (nada em `tests/` importa `bpy`).
