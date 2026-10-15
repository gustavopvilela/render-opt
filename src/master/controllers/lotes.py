from __future__ import annotations
import time
import numpy as np
from fastapi import APIRouter, HTTPException
from comum.protocolo import NovoLote
from comum.util import sha256_arquivo
import master.database as database
import master.planner as planner
from master.config import ConfigMestre
from master.enums import TipoEvento
from master.queries import estimativas as estimativas_sql
from master.queries import eventos as eventos_sql
from master.queries import jobs as jobs_sql
from master.queries import lotes as lotes_sql
from master.queries import trabalhadores as trabalhadores_sql
from master.services.trabalhadores import maquina_de

def criar_router (config: ConfigMestre, db: database.DB) -> APIRouter:
    router = APIRouter()

    @router.post("/lotes")
    def criar_lote (novo: NovoLote) -> dict:
        if len({c.cena_id for c in novo.cenas}) != len(novo.cenas):
            raise HTTPException(422, "Cena repetida no lote")
        cenas, jobs = [], [] # jobs[i] = (cena, frame) da linha i de P
        for c in novo.cenas:
            caminho = config.pasta_cenas / f"{c.cena_id}.blend"
            if not caminho.is_file():
                raise HTTPException(404, f"cena {c.cena_id} nao está em {config.pasta_cenas}")
            if sha256_arquivo(caminho) != c.cena_sha256:
                raise HTTPException(422, f"O SHA-256 da cena {c.cena_id} no mestre difere do informado")
            cenas.append(planner.Cena(id=c.cena_id, caminho=str(caminho), versao=c.versao_cena))
            jobs += [(c, f) for f in sorted(set(c.frames))]

        with db.transacao() as conexao:
            if lotes_sql.existe(conexao, novo.lote_id):
                raise HTTPException(409, "Lote já existe")
            maquinas = [maquina_de(l) for l in trabalhadores_sql.disponiveis(conexao)]

        if not maquinas:
            raise HTTPException(409, "Nenhum trabalhador disponível.")

        matrizes = [planner.estimar_matrizes(cena, sorted(set(c.frames)), maquinas) for cena, c in zip(cenas, novo.cenas)]
        P = np.vstack([p for p, _ in matrizes])
        C = np.vstack([c for _, c in matrizes])

        try:
            plano = planner.gerar_plano_renderizacao(P, C)
        except ValueError as erro:
            raise HTTPException(422, str(erro)) from None

        P = P + C
        if sorted(i for seq in plano for i in seq) != list(range(len(jobs))):
            raise HTTPException(500, "O plano não cobre cada frame exatamente uma vez")

        agora = time.time()
        with db.transacao() as conexao:
            lotes_sql.inserir(conexao, novo.lote_id, agora)

            for k, sequencia in enumerate(plano):
                for posicao, i in enumerate(sequencia, start=1):
                    c, frame = jobs[i]
                    job_id = f"{novo.lote_id}-{c.cena_id}-f{frame:06d}"
                    jobs_sql.inserir(conexao, job_id, novo.lote_id, c.cena_id, c.cena_sha256, c.versao_cena, frame, maquinas[k].id, posicao)
                    estimativas_sql.inserir(
                        conexao,
                        [(job_id, maquinas[c].id, float(P[i, c])) for c in range(len(maquinas)) if np.isfinite(P[i, c])]
                    )

            eventos_sql.registrar_evento(conexao, agora, TipoEvento.LOTE_CRIADO, dados=novo.model_dump_json())

        tempos = [float(sum(P[i, k] for i in seq)) for k, seq in enumerate(plano)]
        return {
            "lote_id": novo.lote_id,
            "jobs": len(jobs),
            "makespan_estimado_s": max(tempos),
            "filas": {maquinas[k].id: len(seq) for k, seq in enumerate(plano)},
        }

    @router.get("/lotes/{lote_id}")
    def status_do_lote (lote_id: str) -> dict:
        with db.transacao() as conexao:
            lote = lotes_sql.buscar(conexao, lote_id)
            if lote is None:
                raise HTTPException(404, "Lote desconhecido")
            por_estado = jobs_sql.contagem_por_estado(conexao, lote_id)

            por_maquina = {}
            for m, n, soma in jobs_sql.concluidos_por_maquina(conexao, lote_id):
                por_maquina[m] = {"concluidos": n, "segundos_renderizando": soma}

            for m, n in jobs_sql.pendentes_por_maquina(conexao, lote_id):
                por_maquina.setdefault(m or "(sem dono)", {})["na fila"] = n

            for m, j in jobs_sql.executando_por_maquina(conexao, lote_id):
                por_maquina.setdefault(m, {})["executando"] = j

        inicio, fim = lote["inicio"], lote["fim"]
        return {
            "lote_id": lote_id,
            "jobs_por_estado": por_estado,
            "por_maquina": por_maquina,
            "concluido": fim is not None,
            "makespan_s": None if inicio is None else (fim or time.time()) - inicio
        }

    return router
