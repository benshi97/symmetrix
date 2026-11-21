#pragma once

#include "../common/neighbor_list.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace mace_polar_ase {

// Host geometry shared by the CPU and CUDA adapters. Candidate image shifts
// remain valid until a skin rebuild; every cell change rebuilds the list.
struct Geometry {
    Geometry(double cutoff, double rmin, const std::vector<int>& elements, double skin)
        : cutoff(cutoff), rmin(rmin), skin(skin), elements(elements) {}

    void update(int n, const double* x, const int* numbers, const double* cell, const bool* pbc) {
        if (n <= 0) throw std::invalid_argument("MACEPolar requires at least one atom");
        if (pbc[0] != pbc[1] or pbc[0] != pbc[2])
            throw std::invalid_argument("MACEPolar requires fully open or fully periodic boundaries");
        for (int k = 0; k < 9; ++k)
            if (not std::isfinite(cell[k])) throw std::invalid_argument("MACEPolar: cell must be finite");
        for (long k = 0; k < 3L*n; ++k)
            if (not std::isfinite(x[k])) throw std::invalid_argument("MACEPolar: positions must be finite");
        const double det = cell[0]*(cell[4]*cell[8]-cell[5]*cell[7])
                         - cell[1]*(cell[3]*cell[8]-cell[5]*cell[6])
                         + cell[2]*(cell[3]*cell[7]-cell[4]*cell[6]);
        if (pbc[0] and (not std::isfinite(det) or det == 0))
            throw std::invalid_argument("MACEPolar requires a nonsingular periodic cell");

        bool rebuild = n != natoms or skin == 0 or pbc[0] != periodic0
            or not std::equal(numbers, numbers + n, numbers0.begin())
            or not std::equal(cell, cell + 9, cell0.begin());
        for (int i = 0; not rebuild and i < n; ++i) {
            double d2 = 0;
            for (int k = 0; k < 3; ++k) {
                const double d = x[3*i+k] - x0[3*i+k];
                d2 += d*d;
            }
            rebuild = 4*d2 > skin*skin;
        }
        if (rebuild) {
            natoms = -1;
            node_types.resize(n);
            for (int i = 0; i < n; ++i) {
                const auto it = std::find(elements.begin(), elements.end(), numbers[i]);
                if (it == elements.end())
                    throw std::invalid_argument("MACEPolar: atomic number " + std::to_string(numbers[i])
                                                + " is not an element of the model");
                node_types[i] = int(it - elements.begin());
            }
            // An open molecule's local and long-range edges are independent of its ASE cell.
            const double no_cell[9]{};
            symmetrix_ase::neighbor_list(n, x, pbc[0] ? cell : no_cell, pbc, cutoff + skin, nl);
            periodic0 = pbc[0];
            numbers0.assign(numbers, numbers + n);
            cell0.assign(cell, cell + 9);
            x0.assign(x, x + 3L*n);
            natoms = n;
            ++builds;
        }

        num_neigh.assign(n, 0);
        neigh_indices.clear();
        neigh_types.clear();
        centers.clear();
        xyz.clear();
        r.clear();
        for (int i = 0; i < n; ++i)
            for (int e = nl.start[i]; e < nl.start[i + 1]; ++e) {
                const int j = nl.j[e];
                const int* s = nl.S.data() + 3L*e;
                double d[3];
                double d2 = 0;
                for (int k = 0; k < 3; ++k) {
                    d[k] = x[3*j+k] - x[3*i+k] + s[0]*cell[k] + s[1]*cell[3+k] + s[2]*cell[6+k];
                    d2 += d[k]*d[k];
                }
                if (d2 >= cutoff*cutoff) continue;
                const double distance = std::sqrt(d2);
                if (distance < rmin)
                    throw std::invalid_argument("MACEPolar: a pair is closer than the model's r_min");
                ++num_neigh[i];
                centers.push_back(i);
                neigh_indices.push_back(j);
                neigh_types.push_back(node_types[j]);
                xyz.insert(xyz.end(), d, d + 3);
                r.push_back(distance);
            }
    }

    double cutoff, rmin, skin;
    std::vector<int> elements;
    int natoms = -1;
    bool periodic0 = false;
    int builds = 0;
    std::vector<double> x0, cell0;
    std::vector<int> numbers0;
    symmetrix_ase::NeighborList nl;
    std::vector<int> node_types, num_neigh, centers, neigh_indices, neigh_types;
    std::vector<double> xyz, r;
};

struct Evaluator {
    virtual ~Evaluator() = default;
    std::mutex mutex;
    virtual double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                           double charge, double spin, double* forces, double* virial, double* energies) = 0;
    virtual int neighbor_list_builds() const = 0;
};

std::unique_ptr<Evaluator> cpu(const std::string& file, double skin);
std::unique_ptr<Evaluator> cuda(const std::string& file, double skin, bool mixed);

}  // namespace mace_polar_ase
