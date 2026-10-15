#include "flamenco.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

namespace Flamenco {
    Tarefas tarefasUnitarias (int jobs) {
        Tarefas tarefas(jobs);
        for (int j = 0; j < jobs; ++j)
            tarefas[j] = {j};
        return tarefas;
    }

    namespace {
        double tempoDaTarefa (const std::vector<int>& tarefa, int maquina, int anterior, const Instancia& instancia) {
            double tempo = 0.0;
            for (int job : tarefa) {
                tempo += anterior >= 0 ? instancia.tempoSetup(maquina, anterior, job) : instancia.tempoSetupInicial(maquina, job);
                tempo += instancia.tempoProcessamento(job, maquina);
                anterior = job;
            }
            return tempo;
        }
    }

    ResultadoFlamenco simular (const Instancia& instancia, const Tarefas& tarefas, const ParametrosFlamenco& parametros) {
        const int m = instancia.maquinas;
        ResultadoFlamenco res;
        res.plano.assign(m, {});
        res.maquina.assign(instancia.jobs, -1);
        res.inicio.assign(instancia.jobs, 0.0);
        res.inicioProcessamento.assign(instancia.jobs, 0.0);
        res.fim.assign(instancia.jobs, 0.0);
        res.temposMaquinas.assign(m, 0.0);

        std::vector<double> livre(m, parametros.esperaInicial);
        std::vector<int> ultimo(m, -1);
        std::vector<char> ativa(m, 1);
        std::vector<char> entregue(tarefas.size(), 0);
        std::size_t primeira = 0;
        std::size_t restantes = tarefas.size();

        while (restantes > 0) {
            int k = -1;
            for (int i = 0; i < m; ++i) {
                if (ativa[i] && (k < 0 || livre[i] < livre[k]))
                    k = i;
            }
            if (k < 0) {
                while (entregue[primeira])
                    ++primeira;
                throw std::invalid_argument(std::format("A tarefa {} (job {}) não tem nenhuma máquina elegível", primeira, tarefas[primeira].front()));
            }

            while (primeira < tarefas.size() && entregue[primeira])
                ++primeira;

            std::size_t escolhida = tarefas.size();
            for (std::size_t t = primeira; t < tarefas.size(); ++t) {
                if (!entregue[t] && std::isfinite(tempoDaTarefa(tarefas[t], k, ultimo[k], instancia))) {
                    escolhida = t;
                    break;
                }
            }
            if (escolhida == tarefas.size()) {
                ativa[k] = 0;
                continue;
            }

            double t = livre[k];
            for (int job : tarefas[escolhida]) {
                const double setup = ultimo[k] >= 0 ? instancia.tempoSetup(k, ultimo[k], job) : instancia.tempoSetupInicial(k, job);
                res.maquina[job] = k;
                res.inicio[job] = t;
                res.inicioProcessamento[job] = t + setup;
                t += setup + instancia.tempoProcessamento(job, k);
                res.fim[job] = t;
                res.plano[k].push_back(job);
                ultimo[k] = job;
            }
            res.temposMaquinas[k] = t;
            livre[k] = t + parametros.intervaloEntreTarefas;
            entregue[escolhida] = 1;
            --restantes;
        }

        for (double tempo : res.temposMaquinas)
            res.makespan = std::max(res.makespan, tempo);
        return res;
    }
}
