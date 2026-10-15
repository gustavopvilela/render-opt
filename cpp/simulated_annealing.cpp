#include "simulated_annealing.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace SimulatedAnnealing {
    JavaRandom::JavaRandom(std::int64_t semente)
        : estado((static_cast<std::uint64_t>(semente) ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1)) {}

    int JavaRandom::next(int bits) {
        estado = (estado * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
        return static_cast<int>(estado >> (48 - bits));
    }

    int JavaRandom::nextInt(int limite) {
        int r = next(31);
        const int m = limite - 1;
        if ((limite & m) == 0)
            return static_cast<int>((static_cast<std::int64_t>(limite) * r) >> 31);
        for (int u = r; static_cast<std::int64_t>(u) - (r = u % limite) + m > std::numeric_limits<int>::max(); u = next(31)) {}
        return r;
    }

    double JavaRandom::nextDouble() {
        const std::int64_t alto = next(26);
        const std::int64_t baixo = next(27);
        return static_cast<double>((alto << 27) + baixo) * 0x1.0p-53;
    }

    double tempoLimiteArtigo(int jobs, int maquinas) {
        return jobs * (maquinas / 2.0) * 50.0 / 2.5 / 1000.0;
    }

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

    namespace {
        int sortearInt (Rng& rng, const int a, const int b) {
            return std::uniform_int_distribution(a, b)(rng);
        }

        /* Solution.java + Machine.java: tempos por máquina atualizados pelo delta de cada operação. */
        class Escalonamento {
        public:
            Escalonamento(const Instancia& instancia, Solucao solucao)
                : instancia(&instancia), seqs(std::move(solucao)), tempos(calcularTemposMaquinas(seqs, instancia)) {
                for (const auto& seq : seqs) {
                    ocupadas += !seq.empty();
                    comDoisJobs += seq.size() > 1;
                }
                atualizarCusto();
            }

            int maquinas() const { return instancia->maquinas; }
            int numJobs(int k) const { return static_cast<int>(seqs[k].size()); }
            int job(int k, int pos) const { return seqs[k][pos]; }
            double makespan() const { return custo; }
            int maquinaMakespan() const { return maquinaDoMakespan; }
            int maquinasOcupadas() const { return ocupadas; }
            int maquinasComDoisJobs() const { return comDoisJobs; }
            const Solucao& sequencias() const { return seqs; }

            /* Solution.updateCost: em empate, fica a primeira máquina; se nenhuma passar de 0, mantém a anterior. */
            double atualizarCusto() {
                custo = 0.0;
                for (int k = 0; k < maquinas(); ++k) {
                    if (tempos[k] > custo) {
                        custo = tempos[k];
                        maquinaDoMakespan = k;
                    }
                }
                return custo;
            }

            double deltaAdicionar(int k, int jobNovo, int pos) const {
                return calcularDeltaInsercao(jobNovo, k, pos, seqs[k], *instancia);
            }

            double deltaRemover(int k, int pos) const {
                const auto& seq = seqs[k];
                const int n = static_cast<int>(seq.size());
                if (n == 1)
                    return -tempos[k];
                const int j = seq[pos];
                const int anterior = pos > 0 ? seq[pos - 1] : -1;
                if (pos == n - 1)
                    return -(entrada(k, anterior, j) + instancia->tempoProcessamento(j, k));
                const int proximo = seq[pos + 1];
                return -(entrada(k, anterior, j) + instancia->tempoProcessamento(j, k) + instancia->tempoSetup(k, j, proximo))
                    + entrada(k, anterior, proximo);
            }

            double deltaTrocar(int k, int jobNovo, int pos) const {
                const auto& seq = seqs[k];
                const int n = static_cast<int>(seq.size());
                const int velho = seq[pos];
                const int anterior = pos > 0 ? seq[pos - 1] : -1;
                const int proximo = pos < n - 1 ? seq[pos + 1] : -1;
                const double antes = entrada(k, anterior, velho) + instancia->tempoProcessamento(velho, k)
                    + (proximo >= 0 ? instancia->tempoSetup(k, velho, proximo) : 0.0);
                const double depois = entrada(k, anterior, jobNovo) + instancia->tempoProcessamento(jobNovo, k)
                    + (proximo >= 0 ? instancia->tempoSetup(k, jobNovo, proximo) : 0.0);
                return -antes + depois;
            }

            void adicionar(int k, int jobNovo, int pos) {
                const double delta = deltaAdicionar(k, jobNovo, pos);
                seqs[k].insert(seqs[k].begin() + pos, jobNovo);
                aplicarDelta(k, delta);
                const int n = numJobs(k);
                if (n == 1) ++ocupadas;
                else if (n == 2) ++comDoisJobs;
            }

            void remover(int k, int pos) {
                const double delta = deltaRemover(k, pos);
                seqs[k].erase(seqs[k].begin() + pos);
                aplicarDelta(k, delta);
                const int n = numJobs(k);
                if (n == 0) --ocupadas;
                else if (n == 1) --comDoisJobs;
            }

            void trocar(int k, int jobNovo, int pos) {
                const double delta = deltaTrocar(k, jobNovo, pos);
                seqs[k][pos] = jobNovo;
                aplicarDelta(k, delta);
            }

        private:
            double entrada(int k, int anterior, int j) const {
                return anterior >= 0 ? instancia->tempoSetup(k, anterior, j) : instancia->tempoSetupInicial(k, j);
            }

            /* Com tempo infinito (máquina inelegível), a soma do delta daria inf - inf; recalcula do zero. */
            void aplicarDelta(int k, double delta) {
                if (std::isfinite(tempos[k]) && std::isfinite(delta))
                    tempos[k] += delta;
                else
                    tempos[k] = calcularTempoMaquina(seqs[k], k, *instancia);
            }

            const Instancia* instancia;
            Solucao seqs;
            std::vector<double> tempos;
            double custo = 0.0;
            int maquinaDoMakespan = 0;
            int ocupadas = 0;
            int comDoisJobs = 0;
        };

        /* Move.java: fazer() aplica o vizinho e devolve o delta do makespan; desfazer() é o reject(). */
        class Movimento {
        public:
            explicit Movimento(bool usarMaquinaMakespan) : mk(usarMaquinaMakespan) {}
            virtual ~Movimento() = default;
            virtual bool temMovimento(const Escalonamento& e) const = 0;
            virtual double fazer(Escalonamento& e, JavaRandom& rng) = 0;
            virtual void desfazer(Escalonamento& e) = 0;

        protected:
            bool mk;
        };

        /*
         * Escolha de máquinas igual à do Java. Onde o Java entraria em laço infinito (nenhuma máquina com dois
         * jobs, uma máquina só, nenhum job), temMovimento devolve false em vez de true.
         */
        bool temMaquinaComDoisJobs(const Escalonamento& e, bool mk) {
            return mk ? e.numJobs(e.maquinaMakespan()) > 1 : e.maquinasComDoisJobs() > 0;
        }

        int escolherMaquinaComDoisJobs(const Escalonamento& e, bool mk, JavaRandom& rng) {
            if (mk && e.numJobs(e.maquinaMakespan()) > 1)
                return e.maquinaMakespan();
            int m;
            do {
                m = rng.nextInt(e.maquinas());
            } while (e.numJobs(m) <= 1);
            return m;
        }

        std::pair<int, int> escolherDuasOcupadas(const Escalonamento& e, bool mk, JavaRandom& rng) {
            const int mm = e.maquinaMakespan();
            if (mk && e.numJobs(mm) > 0) {
                int m;
                do {
                    m = rng.nextInt(e.maquinas());
                } while (m == mm || e.numJobs(m) == 0);
                return {mm, m};
            }
            int m1, m2;
            do {
                m1 = rng.nextInt(e.maquinas());
                m2 = rng.nextInt(e.maquinas());
            } while (m1 == m2 || e.numJobs(m1) == 0 || e.numJobs(m2) == 0);
            return {m1, m2};
        }

        std::pair<int, int> escolherOrigemDestino(const Escalonamento& e, bool mk, JavaRandom& rng) {
            const int mm = e.maquinaMakespan();
            if (mk && e.numJobs(mm) > 0) {
                int m;
                do {
                    m = rng.nextInt(e.maquinas());
                } while (m == mm);
                return {mm, m};
            }
            int m1, m2;
            do {
                m1 = rng.nextInt(e.maquinas());
                m2 = rng.nextInt(e.maquinas());
            } while (m1 == m2 || e.numJobs(m1) == 0);
            return {m1, m2};
        }

        constexpr double INF = std::numeric_limits<double>::infinity();

        /* Shift.java (seção 3.1, política regular) */
        class Shift final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return temMaquinaComDoisJobs(e, mk); }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                maquina = escolherMaquinaComDoisJobs(e, mk, rng);
                do {
                    pos1 = rng.nextInt(e.numJobs(maquina));
                    pos2 = rng.nextInt(e.numJobs(maquina));
                } while (pos1 == pos2);
                job = e.job(maquina, pos1);
                e.remover(maquina, pos1);
                e.adicionar(maquina, job, pos2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.remover(maquina, pos2);
                e.adicionar(maquina, job, pos1);
                e.atualizarCusto();
            }

        private:
            int maquina = 0, pos1 = 0, pos2 = 0, job = 0;
        };

        /* ShiftSmart.java (seção 3.1, intensificação) */
        class ShiftSmart final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return temMaquinaComDoisJobs(e, mk); }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                maquina = escolherMaquinaComDoisJobs(e, mk, rng);
                pos1 = rng.nextInt(e.numJobs(maquina));
                job = e.job(maquina, pos1);
                e.remover(maquina, pos1);

                pos2 = rng.nextInt(e.numJobs(maquina) + 1);
                double custo = INF;
                for (int p = 0; p <= e.numJobs(maquina); ++p) {
                    if (p == pos1) continue;
                    const double simulado = e.deltaAdicionar(maquina, job, p);
                    if (simulado < custo) {
                        custo = simulado;
                        pos2 = p;
                    }
                }
                e.adicionar(maquina, job, pos2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.remover(maquina, pos2);
                e.adicionar(maquina, job, pos1);
                e.atualizarCusto();
            }

        private:
            int maquina = 0, pos1 = 0, pos2 = 0, job = 0;
        };

        /* SimpleSwap.java: "direct swap" (seção 3.6, política regular) */
        class SimpleSwap final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return e.maquinasOcupadas() > 1; }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                std::tie(maquina1, maquina2) = escolherDuasOcupadas(e, mk, rng);
                posM1 = rng.nextInt(e.numJobs(maquina1));
                posM2 = rng.nextInt(e.numJobs(maquina2));
                job1 = e.job(maquina1, posM1);
                job2 = e.job(maquina2, posM2);
                e.trocar(maquina1, job2, posM1);
                e.trocar(maquina2, job1, posM2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.trocar(maquina1, job1, posM1);
                e.trocar(maquina2, job2, posM2);
                e.atualizarCusto();
            }

        private:
            int maquina1 = 0, maquina2 = 0, posM1 = 0, posM2 = 0, job1 = 0, job2 = 0;
        };

        /* SimpleSwapSmart.java: "direct swap" (seção 3.6, intensificação) */
        class SimpleSwapSmart final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return e.maquinasOcupadas() > 1; }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                std::tie(maquina1, maquina2) = escolherDuasOcupadas(e, mk, rng);
                posM1 = rng.nextInt(e.numJobs(maquina1));
                job1 = e.job(maquina1, posM1);

                posM2 = 0;
                job2 = e.job(maquina2, 0);
                double custo = INF;
                for (int p = 0; p < e.numJobs(maquina2); ++p) {
                    const int candidato = e.job(maquina2, p);
                    const double simulado = e.deltaTrocar(maquina1, candidato, posM1) + e.deltaTrocar(maquina2, job1, p);
                    if (simulado < custo) {
                        custo = simulado;
                        posM2 = p;
                        job2 = candidato;
                    }
                }
                e.trocar(maquina1, job2, posM1);
                e.trocar(maquina2, job1, posM2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.trocar(maquina1, job1, posM1);
                e.trocar(maquina2, job2, posM2);
                e.atualizarCusto();
            }

        private:
            int maquina1 = 0, maquina2 = 0, posM1 = 0, posM2 = 0, job1 = 0, job2 = 0;
        };

        /* Swap.java (seção 3.4, política regular) */
        class Swap final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return e.maquinasOcupadas() > 1; }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                std::tie(maquina1, maquina2) = escolherDuasOcupadas(e, mk, rng);
                pos1M1 = rng.nextInt(e.numJobs(maquina1));
                pos1M2 = rng.nextInt(e.numJobs(maquina2));
                pos2M1 = rng.nextInt(e.numJobs(maquina1));
                pos2M2 = rng.nextInt(e.numJobs(maquina2));
                job1 = e.job(maquina1, pos1M1);
                job2 = e.job(maquina2, pos1M2);

                e.remover(maquina1, pos1M1);
                e.remover(maquina2, pos1M2);
                e.adicionar(maquina1, job2, pos2M1);
                e.adicionar(maquina2, job1, pos2M2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.remover(maquina1, pos2M1);
                e.remover(maquina2, pos2M2);
                e.adicionar(maquina1, job1, pos1M1);
                e.adicionar(maquina2, job2, pos1M2);
                e.atualizarCusto();
            }

        private:
            int maquina1 = 0, maquina2 = 0, pos1M1 = 0, pos2M1 = 0, pos1M2 = 0, pos2M2 = 0, job1 = 0, job2 = 0;
        };

        /* SwapSmart.java (seção 3.4, intensificação) */
        class SwapSmart final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return e.maquinasOcupadas() > 1; }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                std::tie(maquina1, maquina2) = escolherDuasOcupadas(e, mk, rng);
                pos1M1 = rng.nextInt(e.numJobs(maquina1));
                pos1M2 = rng.nextInt(e.numJobs(maquina2));
                job1 = e.job(maquina1, pos1M1);
                job2 = e.job(maquina2, pos1M2);

                e.remover(maquina1, pos1M1);
                e.remover(maquina2, pos1M2);

                pos2M1 = melhorPosicao(e, maquina1, job2);
                pos2M2 = melhorPosicao(e, maquina2, job1);

                e.adicionar(maquina1, job2, pos2M1);
                e.adicionar(maquina2, job1, pos2M2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.remover(maquina1, pos2M1);
                e.remover(maquina2, pos2M2);
                e.adicionar(maquina1, job1, pos1M1);
                e.adicionar(maquina2, job2, pos1M2);
                e.atualizarCusto();
            }

        private:
            static int melhorPosicao(const Escalonamento& e, int maquina, int job) {
                int melhor = 0;
                double custo = INF;
                for (int p = 0; p <= e.numJobs(maquina); ++p) {
                    const double simulado = e.deltaAdicionar(maquina, job, p);
                    if (simulado < custo) {
                        custo = simulado;
                        melhor = p;
                    }
                }
                return melhor;
            }

            int maquina1 = 0, maquina2 = 0, pos1M1 = 0, pos2M1 = 0, pos1M2 = 0, pos2M2 = 0, job1 = 0, job2 = 0;
        };

        /* Switch.java (seção 3.2, política regular) */
        class Switch final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return temMaquinaComDoisJobs(e, mk); }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                maquina = escolherMaquinaComDoisJobs(e, mk, rng);
                pos1 = rng.nextInt(e.numJobs(maquina));
                pos2 = rng.nextInt(e.numJobs(maquina));
                job1 = e.job(maquina, pos1);
                job2 = e.job(maquina, pos2);
                e.trocar(maquina, job2, pos1);
                e.trocar(maquina, job1, pos2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.trocar(maquina, job1, pos1);
                e.trocar(maquina, job2, pos2);
                e.atualizarCusto();
            }

        private:
            int maquina = 0, pos1 = 0, pos2 = 0, job1 = 0, job2 = 0;
        };

        /* SwitchSmart.java (seção 3.2, intensificação) */
        class SwitchSmart final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return temMaquinaComDoisJobs(e, mk); }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                maquina = escolherMaquinaComDoisJobs(e, mk, rng);
                pos1 = rng.nextInt(e.numJobs(maquina));
                job1 = e.job(maquina, pos1);

                pos2 = pos1;
                job2 = job1;
                double custo = INF;
                for (int p = 0; p < e.numJobs(maquina); ++p) {
                    if (p == pos1) continue;
                    const int candidato = e.job(maquina, p);
                    const double simulado = e.deltaTrocar(maquina, candidato, pos1) + e.deltaTrocar(maquina, job1, p);
                    if (simulado < custo) {
                        custo = simulado;
                        pos2 = p;
                        job2 = candidato;
                    }
                }
                e.trocar(maquina, job2, pos1);
                e.trocar(maquina, job1, pos2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.trocar(maquina, job1, pos1);
                e.trocar(maquina, job2, pos2);
                e.atualizarCusto();
            }

        private:
            int maquina = 0, pos1 = 0, pos2 = 0, job1 = 0, job2 = 0;
        };

        /* TaskMove.java (seção 3.3, política regular) */
        class TaskMove final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override {
                return e.maquinas() > 1 && e.maquinasOcupadas() > 0;
            }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                std::tie(maquina1, maquina2) = escolherOrigemDestino(e, mk, rng);
                posM1 = rng.nextInt(e.numJobs(maquina1));
                posM2 = rng.nextInt(e.numJobs(maquina2) + 1);
                job = e.job(maquina1, posM1);
                e.remover(maquina1, posM1);
                e.adicionar(maquina2, job, posM2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.adicionar(maquina1, job, posM1);
                e.remover(maquina2, posM2);
                e.atualizarCusto();
            }

        private:
            int maquina1 = 0, maquina2 = 0, posM1 = 0, posM2 = 0, job = 0;
        };

        /* TaskMoveSmart.java (seção 3.3, intensificação) */
        class TaskMoveSmart final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override {
                return e.maquinas() > 1 && e.maquinasOcupadas() > 0;
            }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                std::tie(maquina1, maquina2) = escolherOrigemDestino(e, mk, rng);
                posM1 = rng.nextInt(e.numJobs(maquina1));
                job = e.job(maquina1, posM1);
                e.remover(maquina1, posM1);

                posM2 = rng.nextInt(e.numJobs(maquina2) + 1);
                double custo = INF;
                for (int p = 0; p <= e.numJobs(maquina2); ++p) {
                    const double simulado = e.deltaAdicionar(maquina2, job, p);
                    if (simulado < custo) {
                        custo = simulado;
                        posM2 = p;
                    }
                }
                e.adicionar(maquina2, job, posM2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.adicionar(maquina1, job, posM1);
                e.remover(maquina2, posM2);
                e.atualizarCusto();
            }

        private:
            int maquina1 = 0, maquina2 = 0, posM1 = 0, posM2 = 0, job = 0;
        };

        /* TwoShift.java (seção 3.5, política regular) */
        class TwoShift final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return temMaquinaComDoisJobs(e, mk); }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                maquina = escolherMaquinaComDoisJobs(e, mk, rng);

                pos1_1 = rng.nextInt(e.numJobs(maquina));
                job1 = e.job(maquina, pos1_1);
                e.remover(maquina, pos1_1);

                pos2_1 = rng.nextInt(e.numJobs(maquina));
                job2 = e.job(maquina, pos2_1);
                e.remover(maquina, pos2_1);

                pos1_2 = rng.nextInt(e.numJobs(maquina) + 1);
                e.adicionar(maquina, job1, pos1_2);

                pos2_2 = rng.nextInt(e.numJobs(maquina) + 1);
                e.adicionar(maquina, job2, pos2_2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.remover(maquina, pos2_2);
                e.remover(maquina, pos1_2);
                e.adicionar(maquina, job2, pos2_1);
                e.adicionar(maquina, job1, pos1_1);
                e.atualizarCusto();
            }

        private:
            int maquina = 0, pos1_1 = 0, pos1_2 = 0, pos2_1 = 0, pos2_2 = 0, job1 = 0, job2 = 0;
        };

        /* TwoShiftSmart.java (seção 3.5, intensificação) */
        class TwoShiftSmart final : public Movimento {
        public:
            using Movimento::Movimento;
            bool temMovimento(const Escalonamento& e) const override { return temMaquinaComDoisJobs(e, mk); }

            double fazer(Escalonamento& e, JavaRandom& rng) override {
                const double inicial = e.makespan();
                maquina = escolherMaquinaComDoisJobs(e, mk, rng);

                pos1_1 = rng.nextInt(e.numJobs(maquina));
                job1 = e.job(maquina, pos1_1);
                e.remover(maquina, pos1_1);
                pos1_2 = melhorPosicao(e, maquina, job1, pos1_1);
                e.adicionar(maquina, job1, pos1_2);

                do {
                    pos2_1 = rng.nextInt(e.numJobs(maquina));
                } while (pos2_1 == pos1_2);
                job2 = e.job(maquina, pos2_1);
                e.remover(maquina, pos2_1);
                pos2_2 = melhorPosicao(e, maquina, job2, pos2_1);
                e.adicionar(maquina, job2, pos2_2);
                return e.atualizarCusto() - inicial;
            }

            void desfazer(Escalonamento& e) override {
                e.remover(maquina, pos2_2);
                e.adicionar(maquina, job2, pos2_1);
                e.remover(maquina, pos1_2);
                e.adicionar(maquina, job1, pos1_1);
                e.atualizarCusto();
            }

        private:
            static int melhorPosicao(const Escalonamento& e, int maquina, int job, int origem) {
                int melhor = origem;
                double custo = INF;
                for (int p = 0; p <= e.numJobs(maquina); ++p) {
                    if (p == origem) continue;
                    const double simulado = e.deltaAdicionar(maquina, job, p);
                    if (simulado < custo) {
                        custo = simulado;
                        melhor = p;
                    }
                }
                return melhor;
            }

            int maquina = 0, pos1_1 = 0, pos1_2 = 0, pos2_1 = 0, pos2_2 = 0, job1 = 0, job2 = 0;
        };

        using Vizinhancas = std::vector<std::unique_ptr<Movimento>>;

        /* Main.createNeighborhoods: mesma ordem, que define o índice sorteado em selectMove. */
        Vizinhancas criarVizinhancas(const std::array<bool, NUM_VIZINHANCAS>& ativas) {
            std::array<std::unique_ptr<Movimento>, NUM_VIZINHANCAS> todas = {
                std::make_unique<Shift>(true), std::make_unique<Shift>(false),
                std::make_unique<ShiftSmart>(true), std::make_unique<ShiftSmart>(false),
                std::make_unique<SimpleSwap>(true), std::make_unique<SimpleSwap>(false),
                std::make_unique<SimpleSwapSmart>(true), std::make_unique<SimpleSwapSmart>(false),
                std::make_unique<Swap>(true), std::make_unique<Swap>(false),
                std::make_unique<SwapSmart>(true), std::make_unique<SwapSmart>(false),
                std::make_unique<Switch>(true), std::make_unique<Switch>(false),
                std::make_unique<SwitchSmart>(true), std::make_unique<SwitchSmart>(false),
                std::make_unique<TaskMove>(true), std::make_unique<TaskMove>(false),
                std::make_unique<TaskMoveSmart>(true), std::make_unique<TaskMoveSmart>(false),
                std::make_unique<TwoShift>(true), std::make_unique<TwoShift>(false),
                std::make_unique<TwoShiftSmart>(true), std::make_unique<TwoShiftSmart>(false),
            };
            Vizinhancas lista;
            for (int i = 0; i < NUM_VIZINHANCAS; ++i) {
                if (ativas[i]) lista.push_back(std::move(todas[i]));
            }
            return lista;
        }

        /* Heuristic.selectMove: sorteia uniformemente até achar uma vizinhança aplicável. */
        Movimento* selecionar(const Vizinhancas& lista, const Escalonamento& e, JavaRandom& rng) {
            const int n = static_cast<int>(lista.size());
            Movimento* mov = lista[rng.nextInt(n)].get();
            while (!mov->temMovimento(e)) {
                if (std::none_of(lista.begin(), lista.end(), [&](const auto& v) { return v->temMovimento(e); }))
                    return nullptr;
                mov = lista[rng.nextInt(n)].get();
            }
            return mov;
        }

        std::int64_t sementeAleatoria() {
            std::random_device rd;
            return static_cast<std::int64_t>((static_cast<std::uint64_t>(rd()) << 32) | rd());
        }
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

    /*
     * SimpleConstructive.randomSolution (seção 4.1): embaralha os jobs e põe cada um no fim de uma máquina
     * sorteada. Adaptação: o sorteio é só entre as máquinas com tempo finito para o job. Quando todas são
     * elegíveis, o sorteio é o mesmo do Java.
     */
    Solucao gerarSolucaoAleatoria(const Instancia& instancia, JavaRandom& rng) {
        std::vector<int> ordem(instancia.jobs);
        std::iota(ordem.begin(), ordem.end(), 0);
        for (int i = instancia.jobs; i > 1; --i) {
            const int j = rng.nextInt(i);
            std::swap(ordem[i - 1], ordem[j]);
        }

        Solucao solucao(instancia.maquinas);
        std::vector<int> elegiveis;
        elegiveis.reserve(instancia.maquinas);
        for (int job : ordem) {
            elegiveis.clear();
            for (int k = 0; k < instancia.maquinas; ++k) {
                if (std::isfinite(instancia.tempoProcessamento(job, k)) && std::isfinite(instancia.tempoSetupInicial(k, job)))
                    elegiveis.push_back(k);
            }
            if (elegiveis.empty())
                throw std::invalid_argument(std::format("Job {} não tem nenhuma máquina com tempo finito", job));
            solucao[elegiveis[rng.nextInt(static_cast<int>(elegiveis.size()))]].push_back(job);
        }
        return solucao;
    }

    /* SA.java (Algoritmo 1 do artigo) */
    ResultadoSimulatedAnnealing simulatedAnnealing(const Instancia &instancia, const ParametrosSimulatedAnnealing &parametros, std::optional<Solucao> solucaoInicial) {
        if (!(parametros.temperaturaInicial > 0.0))
            throw std::invalid_argument("A temperatura inicial deve ser > 0");
        if (!(parametros.taxaResfriamento > 0.0 && parametros.taxaResfriamento < 1.0))
            throw std::invalid_argument("A taxa de resfriamento deve estar em (0, 1)");
        if (parametros.iteracoesPorTemperatura < 1)
            throw std::invalid_argument("As iterações por temperatura devem ser >= 1");
        if (!(parametros.temperaturaMinima > 0.0))
            throw std::invalid_argument("A temperatura mínima deve ser > 0");
        if (parametros.tempoLimiteSegundos && !(*parametros.tempoLimiteSegundos > 0.0))
            throw std::invalid_argument("O tempo limite deve ser > 0");
        if (parametros.maxIteracoes && *parametros.maxIteracoes < 0)
            throw std::invalid_argument("O máximo de iterações deve ser >= 0");
        if (std::none_of(parametros.vizinhancas.begin(), parametros.vizinhancas.end(), [](bool v) { return v; }))
            throw std::invalid_argument("Pelo menos uma vizinhança deve estar ativa");

        using Relogio = std::chrono::steady_clock;
        const auto inicio = Relogio::now();

        JavaRandom rng(parametros.semente ? static_cast<std::int64_t>(*parametros.semente) : sementeAleatoria());
        Escalonamento atual(instancia, solucaoInicial ? std::move(*solucaoInicial) : gerarSolucaoAleatoria(instancia, rng));
        if (!std::isfinite(atual.makespan()))
            throw std::invalid_argument("A solução inicial tem máquina com tempo infinito");

        std::optional<double> tempoLimite = parametros.tempoLimiteSegundos;
        if (!tempoLimite && !parametros.maxIteracoes)
            tempoLimite = tempoLimiteArtigo(instancia.jobs, instancia.maquinas);
        const auto prazo = Relogio::now() + std::chrono::duration_cast<Relogio::duration>(std::chrono::duration<double>(tempoLimite.value_or(0.0)));

        const Vizinhancas vizinhancas = criarVizinhancas(parametros.vizinhancas);
        Escalonamento melhor = atual;

        ResultadoSimulatedAnnealing resultado;
        Historico& historico = resultado.historico;
        double temperatura = parametros.temperaturaInicial;
        long long iteracoes = 0;
        int iteracoesNaTemperatura = 0;

        auto registrar = [&] {
            historico.iteracao.push_back(iteracoes);
            historico.temperatura.push_back(temperatura);
            historico.makespanAtual.push_back(atual.makespan());
            historico.melhorMakespan.push_back(melhor.makespan());
        };
        registrar();

        while (true) {
            if (tempoLimite && Relogio::now() >= prazo) {
                resultado.motivoParada = std::format("tempo limite de {:.3f}s atingido", *tempoLimite);
                break;
            }
            if (parametros.maxIteracoes && iteracoes >= *parametros.maxIteracoes) {
                resultado.motivoParada = std::format("{} iterações atingidas", *parametros.maxIteracoes);
                break;
            }

            Movimento* mov = selecionar(vizinhancas, atual, rng);
            if (!mov) {
                resultado.motivoParada = "nenhuma vizinhança aplicável à solução";
                break;
            }

            const double delta = mov->fazer(atual, rng);
            if (delta < 0) {
                if (atual.makespan() < melhor.makespan()) {
                    melhor = atual;
                    registrar();
                }
            }
            else if (delta == 0) {
            }
            else {
                const double x = rng.nextDouble();
                if (!(x < 1.0 / std::exp(delta / temperatura)))
                    mov->desfazer(atual);
            }

            if (++iteracoesNaTemperatura >= parametros.iteracoesPorTemperatura) {
                iteracoesNaTemperatura = 0;
                temperatura = parametros.taxaResfriamento * temperatura;
                if (temperatura < parametros.temperaturaMinima) {
                    temperatura = parametros.temperaturaInicial;
                    ++resultado.reaquecimentos;
                }
            }
            ++iteracoes;
        }

        resultado.melhorSolucao = melhor.sequencias();
        const std::vector<double> tempos = calcularTemposMaquinas(resultado.melhorSolucao, instancia);
        resultado.melhorMakespan = *std::max_element(tempos.begin(), tempos.end());
        resultado.iteracoes = iteracoes;
        resultado.tempoSegundos = std::chrono::duration<double>(Relogio::now() - inicio).count();
        return resultado;
    }
}
