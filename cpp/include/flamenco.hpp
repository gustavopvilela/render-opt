#pragma once

#include <vector>
#include "simulated_annealing.hpp"

/* Distribuição do Flamenco (FindRunnableTask): list scheduling de Graham (1966), sem modelo de custo. */
namespace Flamenco {
    using SimulatedAnnealing::Instancia;
    using SimulatedAnnealing::Solucao;

    /* Cada tarefa é um chunk: jobs executados em sequência pela mesma máquina. */
    using Tarefas = std::vector<std::vector<int>>;

    /* No trabalhador do Flamenco: 3 s e 5 s. */
    struct ParametrosFlamenco {
        double esperaInicial = 0.0;
        double intervaloEntreTarefas = 0.0;
    };

    struct ResultadoFlamenco {
        Solucao plano;
        std::vector<int> maquina;
        std::vector<double> inicio;
        std::vector<double> inicioProcessamento;
        std::vector<double> fim;
        std::vector<double> temposMaquinas;
        double makespan = 0.0;
    };

    Tarefas tarefasUnitarias (int jobs);

    /* A máquina livre mais cedo (empate: menor índice) pega a primeira tarefa da fila com tempo finito nela. */
    ResultadoFlamenco simular (const Instancia& instancia, const Tarefas& tarefas, const ParametrosFlamenco& parametros);
}
