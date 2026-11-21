#pragma once

#include <vector>

namespace symmetrix_ase {
struct NeighborList { std::vector<int> start, j, S; };
void neighbor_list(int n, const double* x, const double* cell, const bool* pbc,
                   double cutoff, NeighborList& nl);
}  // namespace symmetrix_ase
