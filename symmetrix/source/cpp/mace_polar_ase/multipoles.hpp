#pragma once

namespace mace_polar_ase {

#ifdef __CUDACC__
#define MULTIPOLE_HD __host__ __device__
#else
#define MULTIPOLE_HD
#endif

MULTIPOLE_HD inline long moment_row(int atom, int lm, int n) {
#ifdef __CUDA_ARCH__
    // CUDA keeps all scalars first, then three vector components per atom.
    return lm == 0 ? atom : n + 3L*atom + (lm-1);
#else
    return 4L*atom + lm;
#endif
}

template <class T>
MULTIPOLE_HD inline double atomic_dipole(const T* rho, const double* positions,
                                      int n, int atom, int axis) {
    // The native engine uses inverted odd-l multipoles in (y,z,x) order.
    const int lm = (axis+2)%3 + 1;
    return positions[3L*atom+axis]*rho[moment_row(atom, 0, n)] - rho[moment_row(atom, lm, n)];
}

#undef MULTIPOLE_HD
}  // namespace mace_polar_ase
