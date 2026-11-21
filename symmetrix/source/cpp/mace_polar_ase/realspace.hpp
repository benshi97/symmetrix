#pragma once

#include <cmath>
#include <vector>

namespace mace_polar_ase::realspace {

// The graph_longrange finite-difference model uses four sites per atom in
// spherical-harmonic order (origin, y, z, x), with distinct energy/field offsets.
struct Projection {
    double step, inverse_width, coupling, scalar, vector, self[4];
};

enum class Operation { field, reverse_field, energy };

template <class Model, class Run>
void set_hooks(Model& model, bool open, Run run) {
    if (!open) {
        model.features_hook = {};
        model.features_reverse_hook = {};
        model.coulomb_hook = {};
        return;
    }
    model.features_hook = [run](const auto* q, auto* v) {
        run.template operator()<Operation::field>(q, nullptr, v, nullptr);
    };
    model.features_reverse_hook = [run](const auto* q, const auto* v, auto* a, double* f, double*) {
        run.template operator()<Operation::reverse_field>(q, v, a, f);
    };
    model.coulomb_hook = [run](const auto* q, auto* psi, double* f, double*) {
        run.template operator()<Operation::energy>(q, nullptr, psi, f);
    };
}

template <class Model>
std::vector<Projection> projections(const Model& model) {
    constexpr double pi = 3.14159265358979323846;
    const double coupling = model.field_constant / (4*pi);
    const double sigma = model.density_sigma;
    const int widths = int(model.feature_sigmas.size());
    std::vector<Projection> result;
    for (int g = 0; g < widths; ++g) {
        const double width = model.feature_sigmas[g];
        Projection p{0.1, 1/std::sqrt(2*(sigma*sigma + width*width)), coupling,
                     model.feature_a[g][0]/model.density_a[0],
                     model.feature_a[g][1]/model.density_a[1]/0.1, {}};
        // The common field code subtracts the on-site term. Supply it here
        // so that only distinct atoms contribute to the open-boundary field.
        for (int k = 0; k < 4; ++k) p.self[k] = model.field_self[k*widths+g];
        result.push_back(p);
    }
    const double self0 = coupling/(std::sqrt(pi)*sigma);
    const double self1 = self0/(6*sigma*sigma);
    result.push_back({0.02, 0.5/sigma, coupling, 1, 1/0.02, {self0,self1,self1,self1}});
    return result;
}

#ifdef __CUDACC__
#define POLAR_HD __host__ __device__
#else
#define POLAR_HD
#endif

POLAR_HD inline long moment_row(int atom, int lm, int n) {
#ifdef __CUDA_ARCH__
    // The CUDA engine stores the scalar block first, then three vectors per atom.
    return lm == 0 ? atom : n + 3L*atom + (lm-1);
#else
    return 4L*atom + lm;
#endif
}

template <class T>
POLAR_HD inline void spread(const T* moments, int atom, int n, int stride, double scalar, double vector,
                            double (&sites)[4]) {
    sites[0] = scalar * moments[moment_row(atom, 0, n)*stride];
    for (int k = 1; k < 4; ++k) {
        sites[k] = vector * moments[moment_row(atom, k, n)*stride];
        sites[0] -= sites[k];
    }
}

POLAR_HD inline double sum_lanes(double value) {
#ifdef __CUDA_ARCH__
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
#endif
    return value;
}

// One CPU iteration or CUDA warp owns a receiving atom. The transpose and
// coordinate derivative use the same pair kernel as the forward projection.
// No pair table is stored. Each owner writes only its own forces and moments.
template <Operation op, class T>
POLAR_HD void evaluate(int i, int lane, int lanes, int n, const double* positions,
                       const Projection* params, int widths, const T* q,
                       const T* adj, T* output, double* forces) {
    constexpr bool energy = op == Operation::energy;
    constexpr bool reverse = op == Operation::reverse_field;
    constexpr int spins = energy ? 1 : 2;
    double force[3]{};
    for (int g = 0; g < (energy ? 1 : widths); ++g) {
        const auto& p = params[energy ? widths : g];
        const int field_stride = spins * widths;
        double sources_i[spins][4]{}, adj_i[spins][4]{}, potential[spins][4]{};
        for (int s = 0; s < spins; ++s) {
            spread(q+s, i, n, spins, 1, 1/p.step, sources_i[s]);
            if constexpr (reverse)
                spread(adj+s*widths+g, i, n, field_stride,
                       p.scalar, p.vector, adj_i[s]);
        }
        for (int j = lane; j < n; j += lanes) {
            if (j == i) continue;
            double sources_j[spins][4], adj_j[spins][4];
            for (int s = 0; s < spins; ++s) {
                spread(q+s, j, n, spins, 1, 1/p.step, sources_j[s]);
                if constexpr (reverse)
                    spread(adj+s*widths+g, j, n, field_stride,
                           p.scalar, p.vector, adj_j[s]);
            }
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) {
                    double d[3], r2 = 0;
                    for (int k = 0; k < 3; ++k) {
                        const int site = (k+2)%3+1; // x,y,z -> sites 3,1,2
                        d[k] = positions[3L*i+k] - positions[3L*j+k]
                             + ((a == site) - (b == site))*p.step;
                        r2 += d[k]*d[k];
                    }
                    const double r = sqrt(r2);
                    const double denominator = r + 1e-6; // graph_longrange regularizer
                    const double u = p.inverse_width*r;
                    const double kernel = p.coupling*erf(u)/denominator;
                    double weight = 0;
                    for (int s = 0; s < spins; ++s) {
                        if constexpr (reverse) {
                            potential[s][a] += kernel*adj_j[s][b];
                            weight += adj_i[s][a]*sources_j[s][b] + sources_i[s][a]*adj_j[s][b];
                        } else {
                            potential[s][a] += kernel*sources_j[s][b];
                            if constexpr (energy) weight += sources_i[s][a]*sources_j[s][b];
                        }
                    }
                    if constexpr (reverse || energy) {
                        if (r > 0) {
                            constexpr double two_over_sqrt_pi = 1.1283791670955126;
                            const double minus_derivative = (kernel - p.coupling*two_over_sqrt_pi
                                *p.inverse_width*exp(-u*u))/(denominator*r);
                            for (int k = 0; k < 3; ++k) force[k] += weight*minus_derivative*d[k];
                        }
                    }
                }
        }
        for (int s = 0; s < spins; ++s)
            for (int k = 0; k < 4; ++k) {
                const long m = moment_row(i, k, n)*spins+s;
                const long field = m*widths+g;
                const double scalar = reverse ? 1 : p.scalar;
                const double vector = reverse ? 1/p.step : p.vector;
                double value = sum_lanes(k ? vector*(potential[s][k]-potential[s][0])
                                           : scalar*potential[s][0]);
                if (lane == 0) {
                    if constexpr (energy) output[m] = T(0.5*(value + p.self[k]*q[m]));
                    else if constexpr (reverse) output[m] += T(value + p.self[k]*adj[field]);
                    else output[field] = T(value + p.self[k]*q[m]);
                }
            }
    }
    if constexpr (reverse || energy)
        for (int k = 0; k < 3; ++k) {
            const double f = sum_lanes(force[k]);
            if (lane == 0) forces[3L*i+k] += f;
        }
}
#undef POLAR_HD
}  // namespace mace_polar_ase::realspace
