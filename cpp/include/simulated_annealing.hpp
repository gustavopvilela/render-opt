#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace SimulatedAnnealing {
    using Solucao = std::vector<std::vector<int>>;

    struct Instancia {
        int jobs;
        int maquinas;
        const double* processamento;
        const double* setup;

        double tempoProcessamento(int job, int maquina) const {
            return processamento[static_cast<std::size_t>(job) * maquinas + maquina];
        }

        double tempoSetup(int maquina, int anterior, int proximo) const {
            return setup[(static_cast<std::size_t>(maquina) * jobs + anterior) * jobs + proximo];
        }
    };

    struct ParametrosSimulatedAnnealing {
        double temperaturaInicial = 1000.0;
        double taxaResfriamento = 0.99;
        int iteracoesPorTemperatura = 100;
        double temperaturaFinal = 1.0;
        double probAplicarMovimentoInteligente = 0.7;
        std::optional<double> tempoLimiteSegundos;
        std::optional<long long> maxIteracoesSemMelhoria;
        std::optional<std::uint64_t> semente;
        long long intervaloHistorico = 1;
    };

    struct Historico {
        std::vector<long long> iteracao;
        std::vector<double> temperatura;
        std::vector<double> makespanAtual;
        std::vector<double> melhorMakespan;
    };

    struct ResultadoSimulatedAnnealing {
        Solucao melhorSolucao;
        double melhorMakespan = 0.0;
        Historico historico;
        long long iteracoes = 0;
        double tempoSegundos = 0.0;
        std::string motivoParada;
    };

    double calcularTempoMaquina (const std::vector<int>& sequencia, int maquina, const Instancia& instancia);
    std::vector<double> calcularTemposMaquinas (const Solucao& solucao, const Instancia& instancia);
    Solucao gerarSolucaoInicial (const Instancia& instancia);
    ResultadoSimulatedAnnealing simulatedAnnealing (const Instancia& instancia, const ParametrosSimulatedAnnealing& parametros, std::optional<Solucao> solucaoInicial);
}