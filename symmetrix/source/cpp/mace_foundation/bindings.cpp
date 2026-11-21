#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "evaluator.hpp"

namespace py = pybind11;

template <class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(_mace_foundation, m)
{
    py::class_<Evaluator>(m, "MACEFoundation")
        .def(py::init([](const std::string& file, const std::string& device, const std::string& dtype,
                         double skin) -> std::unique_ptr<Evaluator> {
                 if (dtype != "float64" and dtype != "float32" and dtype != "float32-mixed")
                     throw std::invalid_argument("dtype must be 'float64', 'float32' or 'float32-mixed', "
                                                 "not '" + dtype + "'");
                 if (device == "cpu") {
                     if (dtype == "float32-mixed")
                         throw std::invalid_argument("dtype must be 'float64' or 'float32' on the CPU");
                     return mace_foundation_cpu(file, dtype, skin);
                 }
                 if (device != "cuda")
                     throw std::invalid_argument("device must be 'cpu' or 'cuda', not '" + device + "'");
#ifdef MACE_FOUNDATION_CUDA
                 return mace_foundation_cuda(file, dtype, skin);
#else
                 throw std::runtime_error("symmetrix was built without CUDA");
#endif
             }),
             py::arg("model_file"), py::arg("device") = "cuda", py::arg("dtype") = "float64", py::arg("skin") = 1.0)
        .def("compute",
             [](Evaluator& self, Array<double> x, Array<int> numbers, Array<double> cell,
                Array<bool> pbc, bool energies) {
            const int n = int(numbers.size());
            if (x.size() != 3*n or cell.size() != 9 or pbc.size() != 3)
                throw std::invalid_argument("expected x[n][3], numbers[n], cell[3][3], pbc[3]");
            Array<double> forces({n, 3});
            Array<double> virial(6);
            Array<double> e(energies ? n : 0);
            double energy;
            {
                py::gil_scoped_release release;
                energy = self.compute(n, x.data(), numbers.data(), cell.data(), pbc.data(), forces.mutable_data(),
                                      virial.mutable_data(),
                                      energies ? e.mutable_data() : nullptr);
            }
            return py::make_tuple(energy, forces, virial, energies ? py::object(e) : py::none());
        }, py::arg("x"), py::arg("numbers"), py::arg("cell"), py::arg("pbc"), py::arg("energies") = false);

    m.def("neighbor_list", [](Array<double> x, Array<double> cell, Array<bool> pbc, double cutoff) {
        const int n = int(x.size()/3);
        mace_foundation::NeighborList nl;
        mace_foundation::neighbor_list(n, x.data(), cell.data(), pbc.data(), cutoff, nl);
        const long p = long(nl.j.size());
        Array<int> i(p);
        Array<int> j(p);
        Array<int> S({p, 3L});
        for (int a = 0; a < n; ++a)
            for (int q = nl.start[a]; q < nl.start[a + 1]; ++q) i.mutable_data()[q] = a;
        std::copy(nl.j.begin(), nl.j.end(), j.mutable_data());
        std::copy(nl.S.begin(), nl.S.end(), S.mutable_data());
        return py::make_tuple(i, j, S);
    });
}
