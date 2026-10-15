#include "simulated_annealing.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <random>
#include <numeric>
#include <stdexcept>

namespace SimulatedAnnealing {
    double calcularTempoMaquina(const std::vector<int> &sequencia, int maquina, const Instancia &instancia) {
        double tempo = 0.0;
        int anterior = -1;
        for (int job : sequencia) {
            tempo += anterior >= 0 ? instancia.tempoSetup(maquina, anterior, job) : instancia.tempoSetupInicial(maquina, job);
            tempo += instancia.tempoProcessamento(job, maquina);
            anterior = job;
        }
        return tempo;
    }

    std::vector<double> calcularTemposMaquinas(const Solucao &solucao, const Instancia &instancia) {
        std::vector<double> tempos(instancia.maquinas);
        for (int k = 0; k < instancia.maquinas; ++k) {
            tempos[k] = calcularTempoMaquina(solucao[k], k, instancia);
        }
        return tempos;
    }

    /* Operações de vizinhança */
    namespace {
        enum class TipoMovimento { Shift, Swap };

        struct Movimento {
            TipoMovimento tipo;
            int maquinaA;
            int idxA;
            int maquinaB;
            int idxB;
        };

        int sortearInt (Rng& rng, const int a, const int b) {
            return std::uniform_int_distribution(a, b)(rng);
        }

        double sortearReal (Rng& rng) {
            return std::uniform_real_distribution(0.0, 1.0)(rng);
        }

        bool aplicarVizinho (
            Solucao& solucao,
            const std::vector<double>& tempos,
            const Instancia& instancia,
            double taxa,
            Rng& rng,
            std::vector<int>& ocupadas,
            Movimento& movimento
        ) {
            const int maisDemorada = static_cast<int>(std::max_element(tempos.begin(), tempos.end()) - tempos.begin());
            const int menosDemorada = static_cast<int>(std::min_element(tempos.begin(), tempos.end()) - tempos.begin());
            if (solucao[maisDemorada].empty()) return false;

            const bool shift = sortearInt(rng, 0, 1) == 0;
            const bool usarMovimentoInteligente = sortearReal(rng) < taxa;

            ocupadas.clear();
            for (int k = 0; k < instancia.maquinas; ++k) {
                if (!solucao[k].empty()) ocupadas.push_back(k);
            }
            auto escolherOcupada = [&] {
                return ocupadas[sortearInt(rng, 0, static_cast<int>(ocupadas.size()) - 1)];
            };

            if (shift) {
                int origem, destino;
                if (usarMovimentoInteligente && maisDemorada != menosDemorada) {
                    origem = maisDemorada;
                    destino = menosDemorada;
                }
                else {
                    origem = escolherOcupada();
                    destino = sortearInt(rng, 0, instancia.maquinas - 1);
                }
                auto& seqOrigem = solucao[origem];
                const int idx = sortearInt(rng, 0, static_cast<int>(seqOrigem.size()) - 1);
                const int job = seqOrigem[idx];
                seqOrigem.erase(seqOrigem.begin() + idx);

                auto& seqDestino = solucao[destino];
                const int pos = sortearInt(rng, 0, static_cast<int>(seqDestino.size()));
                seqDestino.insert(seqDestino.begin() + pos, job);
                movimento = {TipoMovimento::Shift, origem, idx, destino, pos};
            }
            else {
                int maquina1, maquina2;
                if (usarMovimentoInteligente && ocupadas.size() > 1) {
                    maquina1 = maisDemorada;
                    do {
                        maquina2 = escolherOcupada();
                    } while (maquina1 == maquina2);
                }
                else {
                    maquina1 = escolherOcupada();
                    maquina2 = escolherOcupada();
                }

                const int idx1 = sortearInt(rng, 0, static_cast<int>(solucao[maquina1].size()) - 1);
                const int idx2 = sortearInt(rng, 0, static_cast<int>(solucao[maquina2].size()) - 1);
                std::swap(solucao[maquina1][idx1], solucao[maquina2][idx2]);
                movimento = {TipoMovimento::Swap, maquina1, idx1, maquina2, idx2};
            }

            return true;
        }

        void desfazer (Solucao& solucao, const Movimento& mov) {
            if (mov.tipo == TipoMovimento::Shift) {
                auto& destino = solucao[mov.maquinaB];
                const int job = destino[mov.idxB];
                destino.erase(destino.begin() + mov.idxB);
                auto& origem = solucao[mov.maquinaA];
                origem.insert(origem.begin() + mov.idxA, job);
            }
            else {
                std::swap(solucao[mov.maquinaA][mov.idxA], solucao[mov.maquinaB][mov.idxB]);
            }
        }
    }

    double calcularDeltaInsercao(int job, int maquina, int pos, const std::vector<int> &sequencia, const Instancia &instancia) {
        const int anterior = pos > 0 ? sequencia[pos - 1] : -1;
        const int proximo = pos < static_cast<int>(sequencia.size()) ? sequencia[pos] : -1;
        double delta = instancia.tempoProcessamento(job, maquina);
        delta += anterior >= 0 ? instancia.tempoSetup(maquina, anterior, job) : instancia.tempoSetupInicial(maquina, job);
        if (proximo >= 0) {
            delta += instancia.tempoSetup(maquina, job, proximo);
            delta -= anterior >= 0 ? instancia.tempoSetup(maquina, anterior, proximo) : instancia.tempoSetupInicial(maquina, proximo);
        }
        return delta;
    }

    Solucao gerarSolucaoInicial(const Instancia &instancia, double alpha, Rng& rng) {
        struct Candidato {
            int maquina;
            int pos;
            double makespan;
            double aumento;
        };

        Solucao solucao(instancia.maquinas);
        std::vector<double> tempos(instancia.maquinas, 0.0);
        std::vector<int> jobsNaoAlocados(instancia.jobs);
        std::iota(jobsNaoAlocados.begin(), jobsNaoAlocados.end(), 0);
        std::shuffle(jobsNaoAlocados.begin(), jobsNaoAlocados.end(), rng);
        std::vector<Candidato> custos;
        std::vector<int> rcl;

        for (int job : jobsNaoAlocados) {
            int maisDemorada = 0;
            double maior = -std::numeric_limits<double>::infinity();
            double segundo = maior;

            for (int k = 0; k < instancia.maquinas; ++k) {
                if (tempos[k] > maior) {
                    segundo = maior;
                    maior = tempos[k];
                    maisDemorada = k;
                }
                else if (tempos[k] > segundo) {
                    segundo = tempos[k];
                }
            }

            custos.clear();
            for (int m = 0; m < instancia.maquinas; ++m) {
                const double maiorOutras = (m == maisDemorada) ? segundo : maior;
                for (int pos = 0; pos <= static_cast<int>(solucao[m].size()); ++pos) {
                    const double aumento = calcularDeltaInsercao(job, m, pos, solucao[m], instancia);
                    if (!std::isfinite(aumento)) continue;
                    const double makespanVizinho = std::max(tempos[m] + aumento, maiorOutras);
                    custos.push_back({.maquina = m, .pos = pos, .makespan = makespanVizinho, .aumento = aumento});
                }
            }

            if (custos.empty())
                throw std::invalid_argument(std::format("Job {} não tem nenhuma máquina com tempo finito", job));

            const auto [itMin, itMax] = std::minmax_element(custos.begin(), custos.end(), [](const Candidato& a, const Candidato& b) {
                return a.makespan < b.makespan;
            });
            const double limite = itMin->makespan + alpha * (itMax->makespan - itMin->makespan) + 1e-5;

            rcl.clear();
            for (int i = 0; i < static_cast<int>(custos.size()); ++i) {
                if (custos[i].makespan <= limite) rcl.push_back(i);
            }
            if (rcl.empty()) rcl.push_back(0);

            const Candidato& escolhido = custos[rcl[sortearInt(rng, 0, static_cast<int>(rcl.size()) - 1)]];
            auto& seq = solucao[escolhido.maquina];
            seq.insert(seq.begin() + escolhido.pos, job);
            tempos[escolhido.maquina] += escolhido.aumento;
        }

        return solucao;
    }

    /* Simulated Annealing principal */
    ResultadoSimulatedAnnealing simulatedAnnealing(const Instancia &instancia, const ParametrosSimulatedAnnealing &parametros, std::optional<Solucao> solucaoInicial) {
        using Relogio = std::chrono::steady_clock;
        const auto inicio = Relogio::now();
        auto segundosDecorridos = [&] {
            return std::chrono::duration<double>(Relogio::now() - inicio).count();
        };

        Rng rng(parametros.semente ? *parametros.semente : std::random_device{}());

        Solucao atual = solucaoInicial ? std::move(*solucaoInicial) : gerarSolucaoInicial(instancia, parametros.alphaConstrucao, rng);
        std::vector<double> tempos = calcularTemposMaquinas(atual, instancia);
        double makespanAtual = *std::max_element(tempos.begin(), tempos.end());

        ResultadoSimulatedAnnealing resultado;
        resultado.melhorSolucao = atual;
        resultado.melhorMakespan = makespanAtual;
        resultado.motivoParada = "Temperatura mínima atingida";
        Historico& historico = resultado.historico;

        double T = parametros.temperaturaInicial;
        long long iteracaoGlobal = 0;
        long long iteracoesSemMelhora = 0;
        bool pararBusca = false;

        std::vector<int> ocupadas;
        ocupadas.reserve(instancia.maquinas);
        Movimento mov{};

        while (T > parametros.temperaturaFinal && !pararBusca) {
            for (int r = 0; r < parametros.iteracoesPorTemperatura; ++r) {
                ++iteracaoGlobal;
                if (!aplicarVizinho(atual, tempos, instancia, parametros.probAplicarMovimentoInteligente, rng, ocupadas, mov))
                    continue;

                /* Delta evaluation */
                const double antigoA = tempos[mov.maquinaA];
                const double antigoB = tempos[mov.maquinaB];
                tempos[mov.maquinaA] = calcularTempoMaquina(atual[mov.maquinaA], mov.maquinaA, instancia);
                if (mov.maquinaB != mov.maquinaA)
                    tempos[mov.maquinaB] = calcularTempoMaquina(atual[mov.maquinaB], mov.maquinaB, instancia);

                const double makespanVizinho = *std::max_element(tempos.begin(), tempos.end());
                const double delta = makespanVizinho - makespanAtual;

                const bool aceitar = delta < 0 || sortearReal(rng) < std::exp(-delta / T);
                bool melhorou = false;
                if (aceitar) {
                    makespanAtual = makespanVizinho;
                    if (makespanAtual < resultado.melhorMakespan) {
                        resultado.melhorMakespan = makespanAtual;
                        resultado.melhorSolucao = atual;
                        melhorou = true;
                    }
                }
                else {
                    desfazer(atual, mov);
                    tempos[mov.maquinaA] = antigoA;
                    tempos[mov.maquinaB] = antigoB;
                }

                iteracoesSemMelhora = melhorou ? 0 : iteracoesSemMelhora + 1;
                if (iteracaoGlobal % parametros.intervaloHistorico == 0) {
                    historico.iteracao.emplace_back(iteracaoGlobal);
                    historico.temperatura.emplace_back(T);
                    historico.makespanAtual.emplace_back(makespanAtual);
                    historico.melhorMakespan.emplace_back(resultado.melhorMakespan);
                }

                if (parametros.tempoLimiteSegundos && segundosDecorridos() >= *parametros.tempoLimiteSegundos) {
                    resultado.motivoParada = std::format("tempo limite de {}s atingido", *parametros.tempoLimiteSegundos);
                    pararBusca = true;
                    break;
                }

                if (parametros.maxIteracoesSemMelhoria && iteracoesSemMelhora >= *parametros.maxIteracoesSemMelhoria) {
                    resultado.motivoParada = std::format("sem melhora por {} iterações consecutivas", *parametros.maxIteracoesSemMelhoria);
                    pararBusca = true;
                    break;
                }
            }

            T *= parametros.taxaResfriamento;
        }

        resultado.iteracoes = iteracaoGlobal;
        resultado.tempoSegundos = segundosDecorridos();
        return resultado;
    }
}
