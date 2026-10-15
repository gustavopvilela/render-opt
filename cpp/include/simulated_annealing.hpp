#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

/*
 * Simulated Annealing de Santos et al. (2016), "Analysis of stochastic local search methods for the
 * unrelated parallel machine scheduling problem", ITOR, DOI 10.1111/itor.12316 (Algoritmo 1 e seção 3).
 * Tradução do código Java dos autores (novo-sa/upmsp): SA.java, Heuristic.java, SimpleConstructive.java,
 * Machine.java, Solution.java e as 12 classes de vizinhança.
 */
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

    /* Réplica do java.util.Random, para que a mesma semente gere a mesma busca do código Java. */
    class JavaRandom {
    public:
        explicit JavaRandom(std::int64_t semente);
        int nextInt(int limite);
        double nextDouble();

    private:
        int next(int bits);
        std::uint64_t estado;
    };

    /*
     * Índice = vizinhança * 4 + estratégia, como a opção "-n" do Main.java.
     * Vizinhanças: 0 Shift, 1 Direct swap (SimpleSwap), 2 Swap, 3 Switch, 4 Task move, 5 2-Shift.
     * Estratégias: 0 regular + máquina do makespan, 1 regular + máquina aleatória,
     *              2 intensificação + máquina do makespan, 3 intensificação + máquina aleatória.
     */
    constexpr int NUM_VIZINHANCAS = 24;

    /* Valores padrão: Tabela 1 do artigo (irace). A temperatura mínima é o EPS de reaquecimento do SA.java. */
    struct ParametrosSimulatedAnnealing {
        double temperaturaInicial = 1.0;
        double taxaResfriamento = 0.96;
        int iteracoesPorTemperatura = 1'176'628;
        double temperaturaMinima = 1e-6;
        std::optional<double> tempoLimiteSegundos;
        std::optional<long long> maxIteracoes;
        std::optional<std::uint64_t> semente;
        std::array<bool, NUM_VIZINHANCAS> vizinhancas = [] {
            std::array<bool, NUM_VIZINHANCAS> todas{};
            todas.fill(true);
            return todas;
        }();
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
        long long reaquecimentos = 0;
        double tempoSegundos = 0.0;
        std::string motivoParada;
    };

    /* Limite de Vallada e Ruiz (2011), n * (m / 2) * t ms com t = 50, dividido por 2,5 como na seção 5 do artigo. */
    double tempoLimiteArtigo(int jobs, int maquinas);

    double calcularTempoMaquina (const std::vector<int>& sequencia, int maquina, const Instancia& instancia);
    std::vector<double> calcularTemposMaquinas (const Solucao& solucao, const Instancia& instancia);
    double calcularDeltaInsercao (int job, int maquina, int pos, const std::vector<int>& sequencia, const Instancia& instancia);
    Solucao gerarSolucaoInicial (const Instancia& instancia, double alpha, Rng& rng);
    Solucao gerarSolucaoAleatoria (const Instancia& instancia, JavaRandom& rng);
    ResultadoSimulatedAnnealing simulatedAnnealing (const Instancia& instancia, const ParametrosSimulatedAnnealing& parametros, std::optional<Solucao> solucaoInicial);
}
