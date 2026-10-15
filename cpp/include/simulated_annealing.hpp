#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace SimulatedAnnealing {
    using Solucao = std::vector<std::vector<int>>;
    using Rng = std::mt19937_64;

    struct Instancia {
        int jobs;
        int maquinas;
        const double* processamento;
        const double* setup;
        const double* setupInicial = nullptr;

        double tempoProcessamento(int job, int maquina) const {
            return processamento[static_cast<std::size_t>(job) * maquinas + maquina];
        }

        double tempoSetup(int maquina, int anterior, int proximo) const {
            return setup[(static_cast<std::size_t>(maquina) * jobs + anterior) * jobs + proximo];
        }

        double tempoSetupInicial(int maquina, int job) const {
            return setupInicial ? setupInicial[static_cast<std::size_t>(maquina) * jobs + job] : 0.0;
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
        double alphaConstrucao = 0.3;
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
    double calcularDeltaInsercao (int job, int maquina, int pos, const std::vector<int>& sequencia, const Instancia& instancia);
    Solucao gerarSolucaoInicial (const Instancia& instancia, double alpha, Rng& rng);
    ResultadoSimulatedAnnealing simulatedAnnealing (const Instancia& instancia, const ParametrosSimulatedAnnealing& parametros, std::optional<Solucao> solucaoInicial);
}