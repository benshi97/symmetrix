#include "evaluator.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <limits>
#include <mutex>

namespace py = pybind11;
template <class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(_mace_polar, m) {
    using mace_polar_ase::Evaluator;
    py::class_<Evaluator>(m, "MACEPolar")
        .def(py::init([](const std::string& file, const std::string& device,
                         const std::string& dtype, double skin) {
            if (not std::isfinite(skin) or skin < 0)
                throw std::invalid_argument("skin must be finite and nonnegative");
            if (device == "cpu") {
                if (dtype != "float64") throw std::invalid_argument("PolarMACE CPU supports dtype='float64' only");
                return mace_polar_ase::cpu(file, skin);
            }
            if (device != "cuda") throw std::invalid_argument("device must be 'cpu' or 'cuda'");
            if (dtype != "float32" and dtype != "float32-mixed")
                throw std::invalid_argument("PolarMACE CUDA supports 'float32' or 'float32-mixed'; use CPU for float64");
#ifdef MACE_POLAR_ASE_CUDA
            return mace_polar_ase::cuda(file, skin, dtype == "float32-mixed");
#else
            throw std::runtime_error("symmetrix was built without PolarMACE CUDA support");
#endif
        }), py::arg("model_file"), py::arg("device") = "cuda", py::arg("dtype") = "float32", py::arg("skin") = 1.0)
        .def_property_readonly("neighbor_list_builds", &Evaluator::neighbor_list_builds)
        .def("compute", [](Evaluator& self, Array<double> x, Array<int> numbers, Array<double> cell,
                           Array<bool> pbc, double charge, double spin, bool energies, bool dipole) -> py::tuple {
            if (numbers.ndim() != 1 or numbers.size() > std::numeric_limits<int>::max())
                throw std::invalid_argument("expected numbers[n]");
            const int n = int(numbers.size());
            if (x.ndim() != 2 or x.shape(0) != n or x.shape(1) != 3
                or cell.ndim() != 2 or cell.shape(0) != 3 or cell.shape(1) != 3
                or pbc.ndim() != 1 or pbc.size() != 3)
                throw std::invalid_argument("expected x[n][3], numbers[n], cell[3][3], pbc[3]");
            if (not std::isfinite(charge) or not std::isfinite(spin))
                throw std::invalid_argument("charge and spin must be finite");
            Array<double> f({n, 3});
            Array<double> w(6);
            Array<double> e(energies ? n : 0);
            Array<double> mu(dipole ? 3 : 0);
            double energy;
            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(self.mutex);
                energy = self.compute(n, x.data(), numbers.data(), cell.data(), pbc.data(), charge, spin,
                                      f.mutable_data(), w.mutable_data(), energies ? e.mutable_data() : nullptr,
                                      dipole ? mu.mutable_data() : nullptr);
            }
            py::object atomic_energies = energies ? py::object(e) : py::none();
            if (dipole) return py::make_tuple(energy, f, w, atomic_energies, mu);
            return py::make_tuple(energy, f, w, atomic_energies);
        }, py::arg("x"), py::arg("numbers"), py::arg("cell"), py::arg("pbc"),
           py::arg("charge") = 0.0, py::arg("spin") = 1.0, py::arg("energies") = false,
           py::arg("dipole") = false);
}
