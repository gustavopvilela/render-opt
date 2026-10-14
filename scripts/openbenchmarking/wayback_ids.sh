#!/bin/sh
# Lista os IDs de resultados do OpenBenchmarking.org guardados no Wayback Machine.
# uso: sh wayback_ids.sh pasta_saida   (gera pasta_saida/ids_all.txt)
# Depois: python clone.py ids_all.txt pasta_clones 4
#         python tabela.py saida.csv pasta_clones/blender
set -e
cd "$1"
n=$(curl -s -m 120 "https://web.archive.org/cdx/search/cdx?url=openbenchmarking.org/result/&matchType=prefix&showNumPages=true")
for p in $(seq 0 $((n - 1))); do
  [ -s "p$p.txt" ] && continue
  for t in 1 2 3; do
    curl -s -m 120 "https://web.archive.org/cdx/search/cdx?url=openbenchmarking.org/result/&matchType=prefix&fl=original,timestamp&collapse=urlkey&page=$p" -o "p$p.txt" && [ -s "p$p.txt" ] && break
    sleep 5
  done
done
cat p*.txt | grep -oE 'result/[0-9]{7}-[A-Za-z0-9]+-[A-Za-z0-9]+' | sed 's#result/##' | sort -u > ids_all.txt
wc -l ids_all.txt
