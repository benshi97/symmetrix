#include "evaluator.hpp"
#include "mace_polar.hpp"

#include <numeric>
#include <span>

namespace mace_polar_ase {
namespace {

struct CPU : Evaluator {
    CPU(const std::string& file, double skin)
        : model(file), geometry(model.r_cut, model.radial_spline_r_min, model.atomic_numbers, skin) {}

    double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                   double charge, double spin, double* forces, double* virial, double* energies) override {
        geometry.update(n, x, numbers, cell, pbc);
        model.compute_node_energies_forces(n, geometry.node_types, geometry.num_neigh,
            geometry.neigh_indices, geometry.neigh_types, geometry.xyz, geometry.r,
            std::span<const double>(x, 3L*n), std::span<const double>(cell, 9), charge, spin);
        std::copy_n(model.position_forces.data(), 3L*n, forces);
        for (size_t e = 0; e < geometry.r.size(); ++e)
            for (int k = 0; k < 3; ++k) {
                const double f = model.node_forces[3*e+k];
                forces[3*geometry.centers[e]+k] -= f;
                forces[3*geometry.neigh_indices[e]+k] += f;
            }
        std::copy(model.virial.begin(), model.virial.end(), virial);
        if (energies) std::copy_n(model.node_energies.data(), n, energies);
        return std::accumulate(model.node_energies.begin(), model.node_energies.begin() + n, 0.0);
    }

    int neighbor_list_builds() const override { return geometry.builds; }
    MACEPolar model;
    Geometry geometry;
};

}  // namespace

std::unique_ptr<Evaluator> cpu(const std::string& file, double skin) {
    return std::make_unique<CPU>(file, skin);
}

}  // namespace mace_polar_ase
