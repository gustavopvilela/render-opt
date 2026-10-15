#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <format>
#include <stdexcept>
#include "simulated_annealing.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {
    using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

    SimulatedAnnealing::Instancia lerInstancia (const py::object& obj, ArrayD& proc, ArrayD& setup, ArrayD& inicial) {
        const int jobs = obj.attr("jobs").cast<int>();
        const int maquinas = obj.attr("maquinas").cast<int>();
        proc = obj.attr("matriz_processamento").cast<ArrayD>();
        setup = obj.attr("matriz_setup").cast<ArrayD>();

        if (jobs < 0 || maquinas < 1)
            throw std::invalid_argument("A instância precisa que jobs >= 0 e maquinas >= 1");
        if (proc.ndim() != 2 || proc.shape(0) != jobs || proc.shape(1) != maquinas)
            throw std::invalid_argument(std::format("Matriz de processamento deve ter shape (jobs, maquians) = ({}, {})", jobs, maquinas));
        if (setup.ndim() != 3 || setup.shape(0) != maquinas || setup.shape(1) != jobs || setup.shape(2) != jobs)
            throw std::invalid_argument(std::format("matriz_setup seve ter shape (maquinas, jobs, jobs) = ({}, {}, {})", maquinas, jobs, jobs));

        const py::object objInicial = py::getattr(obj, "setup_inicial", py::none());
        if (objInicial.is_none())
            return {jobs, maquinas, proc.data(), setup.data()};

        inicial = objInicial.cast<ArrayD>();
        if (inicial.ndim() != 2 || inicial.shape(0) != maquinas || inicial.shape(1) != jobs)
            throw std::invalid_argument(std::format("setup_inicial deve ter shape (maquinas, jobs) = ({}, {})", maquinas, jobs));
        return {jobs, maquinas, proc.data(), setup.data(), inicial.data()};
    }

    void validarSolucao (const SimulatedAnnealing::Solucao& solucao, const SimulatedAnnealing::Instancia& instancia) {
        if (static_cast<int>(solucao.size()) != instancia.maquinas)
            throw std::invalid_argument(std::format("A solução deve ter {} listas (uma por máquina)", instancia.maquinas));

        std::vector<char> visto(instancia.jobs, 0);
        for (const auto& seq : solucao) {
            for (int job : seq) {
                if (job < 0 || job >= instancia.jobs)
                    throw std::invalid_argument(std::format("Job {} fora do intervalo [0, {})", job, instancia.jobs));
                if (visto[job]++) {
                    throw std::invalid_argument(std::format("Job {} aparece mais de uma vez", job));
                }
            }
        }

        for (int j = 0; j < instancia.jobs; ++j) {
            if (!visto[j])
                throw std::invalid_argument(std::format("Job {} não foi alocado", j));
        }
    }

    void validarAlpha (double alpha) {
        if (!(alpha >= 0.0 && alpha <= 1.0))
            throw std::invalid_argument("Alpha deve estar em [0, 1]");
    }

    template <typename T>
    py::array_t<T> paraNumpy (std::vector<T>&& v) {
        auto* dono = new std::vector<T>(std::move(v));
        py::capsule liberar(dono, [](void* p) { delete static_cast<std::vector<T>*>(p); });
        return py::array_t<T>(static_cast<py::ssize_t>(dono->size()), dono->data(), liberar);
    }

    py::tuple simulatedAnnealingPy (
        const py::object& instancia,
        double temperaturaInicial,
        double taxaResfriamento,
        int iteracoesPorTemperatura,
        std::optional<double> tempoLimiteSegundos,
        std::optional<long long> maxIteracoes,
        std::optional<std::uint64_t> semente,
        std::optional<SimulatedAnnealing::Solucao> solucaoInicial,
        std::optional<std::vector<bool>> vizinhancas,
        bool verbose
    ) {
        ArrayD processamento, setup, inicial;
        const SimulatedAnnealing::Instancia inst = lerInstancia(instancia, processamento, setup, inicial);
        if (solucaoInicial)
            validarSolucao(*solucaoInicial, inst);

        SimulatedAnnealing::ParametrosSimulatedAnnealing par;
        par.temperaturaInicial = temperaturaInicial;
        par.taxaResfriamento = taxaResfriamento;
        par.iteracoesPorTemperatura = iteracoesPorTemperatura;
        par.tempoLimiteSegundos = tempoLimiteSegundos;
        par.maxIteracoes = maxIteracoes;
        par.semente = semente;
        if (vizinhancas) {
            if (vizinhancas->size() != SimulatedAnnealing::NUM_VIZINHANCAS)
                throw std::invalid_argument(std::format("vizinhancas deve ter {} valores", SimulatedAnnealing::NUM_VIZINHANCAS));
            std::copy(vizinhancas->begin(), vizinhancas->end(), par.vizinhancas.begin());
        }

        SimulatedAnnealing::ResultadoSimulatedAnnealing res;
        {
            py::gil_scoped_release semGIL;
            res = SimulatedAnnealing::simulatedAnnealing(inst, par, std::move(solucaoInicial));
        }

        if (verbose)
            py::print(std::format("Simulated Annealing finalizado ({}) -- {} iterações, {} reaquecimentos em {:.4f}s",
                res.motivoParada, res.iteracoes, res.reaquecimentos, res.tempoSegundos));

        py::dict historico(
            "iteracao"_a = paraNumpy(std::move(res.historico.iteracao)),
            "temperatura"_a = paraNumpy(std::move(res.historico.temperatura)),
            "makespan_atual"_a = paraNumpy(std::move(res.historico.makespanAtual)),
            "melhor_makespan"_a = paraNumpy(std::move(res.historico.melhorMakespan)),
            "iteracoes"_a = res.iteracoes,
            "reaquecimentos"_a = res.reaquecimentos,
            "tempo_segundos"_a = res.tempoSegundos,
            "motivo_parada"_a = res.motivoParada
        );

        return py::make_tuple(res.melhorSolucao, res.melhorMakespan, historico);
    }

    py::array_t<double> calcularTemposMaquinasPy (const py::object& instancia, const SimulatedAnnealing::Solucao& solucao) {
        ArrayD processamento, setup, inicial;
        const SimulatedAnnealing::Instancia inst = lerInstancia(instancia, processamento, setup, inicial);
        validarSolucao(solucao, inst);
        return paraNumpy(SimulatedAnnealing::calcularTemposMaquinas(solucao, inst));
    }

    SimulatedAnnealing::Solucao gerarSolucaoInicialPy (const py::object& instancia, double alpha, std::optional<std::uint64_t> semente) {
        ArrayD processamento, setup, inicial;
        const SimulatedAnnealing::Instancia inst = lerInstancia(instancia, processamento, setup, inicial);
        validarAlpha(alpha);
        SimulatedAnnealing::Rng rng(semente ? *semente : std::random_device{}());
        py::gil_scoped_release semGIL;
        return SimulatedAnnealing::gerarSolucaoInicial(inst, alpha, rng);
    }

    SimulatedAnnealing::Solucao gerarSolucaoAleatoriaPy (const py::object& instancia, std::int64_t semente) {
        ArrayD processamento, setup, inicial;
        const SimulatedAnnealing::Instancia inst = lerInstancia(instancia, processamento, setup, inicial);
        SimulatedAnnealing::JavaRandom rng(semente);
        return SimulatedAnnealing::gerarSolucaoAleatoria(inst, rng);
    }

    double calcularDeltaInsercaoPy (int job, int maquina, int pos, const std::vector<int>& sequencia, const py::object& instancia) {
        ArrayD processamento, setup, inicial;
        const SimulatedAnnealing::Instancia inst = lerInstancia(instancia, processamento, setup, inicial);
        auto fora = [](int x, int fim) { return x < 0 || x >= fim; };

        if (fora(job, inst.jobs))
            throw std::invalid_argument(std::format("job {} fora do intervalo", job));
        if (fora(maquina, inst.maquinas))
            throw std::invalid_argument(std::format("máquina {} fora do intervalo", maquina));
        if (fora(pos, static_cast<int>(sequencia.size()) + 1))
            throw std::invalid_argument(std::format("pos deve estar entre 0 e {}", sequencia.size()));

        for (int j : sequencia) {
            if (fora(j, inst.jobs))
                throw std::invalid_argument(std::format("job {} fora do intervalo", j));
        }

        return SimulatedAnnealing::calcularDeltaInsercao(job, maquina, pos, sequencia, inst);
    }
}

PYBIND11_MODULE(sa_core, m) {
    m.doc() = "Simulated Annealing para o UPMSP com setup dependente da sequência em C++.";

    m.def(
        "simulated_annealing", &simulatedAnnealingPy,
        py::arg("instancia"),
        py::arg("temp_inicial") = 1.0,
        py::arg("taxa_resfriamento") = 0.96,
        py::arg("iteracoes_por_temp") = 1'176'628,
        py::arg("tempo_limite_segundos") = py::none(),
        py::arg("max_iteracoes") = py::none(),
        py::kw_only(),
        py::arg("semente") = py::none(),
        py::arg("solucao_inicial") = py::none(),
        py::arg("vizinhancas") = py::none(),
        py::arg("verbose") = true,
        "SA de Santos et al. (2016). Padrões da Tabela 1 do artigo. Sem tempo_limite_segundos nem max_iteracoes, "
        "para no tempo do artigo (tempo_limite_artigo). Devolve (melhor_solucao, melhor_makespan, historico)."
    );

    m.def("tempo_limite_artigo", &SimulatedAnnealing::tempoLimiteArtigo,
        py::arg("jobs"),
        py::arg("maquinas"),
        "Tempo limite do artigo em segundos: n * (m / 2) * 50 ms / 2,5."
    );

    m.def("gerar_solucao_aleatoria", &gerarSolucaoAleatoriaPy,
        py::arg("instancia"),
        py::arg("semente"),
        "Construção aleatória do artigo (seção 4.1), com o mesmo sorteio do java.util.Random."
    );

    m.def(
        "calcular_tempos_maquinas", &calcularTemposMaquinasPy,
        py::arg("instancia"),
        py::arg("solucao"),
        "Tempo de conclusão de cada máquina para a solução dada."
    );

    m.def("gerar_solucao_inicial", &gerarSolucaoInicialPy,
        py::arg("instancia"),
        py::arg("alpha") = 0.3,
        py::kw_only(),
        py::arg("semente") = py::none(),
        "Construção gulosa aleatorizada do GRASP. Devolve uma solução."
    );

    m.def("calcular_delta_insercao", &calcularDeltaInsercaoPy,
        py::arg("job"),
        py::arg("maquina"),
        py::arg("pos"),
        py::arg("sequencia"),
        py::arg("instancia"),
        "Aumento do tempo da máquina ao inserir \'job\' na posição \'pos\' da sequência."
    );
}