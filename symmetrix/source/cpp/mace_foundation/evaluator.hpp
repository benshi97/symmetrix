#pragma once

#include <memory>
#include <string>
#include <vector>

struct Evaluator {
    virtual ~Evaluator() = default;
    virtual double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                           double* forces, double* virial, double* energies) = 0;
};

std::unique_ptr<Evaluator> mace_foundation_cpu(const std::string& file, const std::string& dtype, double skin);
std::unique_ptr<Evaluator> mace_foundation_cuda(const std::string& file, const std::string& dtype, double skin);

#include "../common/neighbor_list.hpp"

namespace mace_foundation {
using symmetrix_ase::NeighborList;
using symmetrix_ase::neighbor_list;
}  // namespace mace_foundation
