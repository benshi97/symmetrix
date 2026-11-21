#include "evaluator.hpp"
#include "mace_polar_cuda.hpp"
#include "realspace.hpp"

#include <cub/block/block_reduce.cuh>

#include <numeric>
#include <span>

namespace mace_polar_ase {
namespace {

void check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("MACEPolar CUDA: ") + cudaGetErrorString(error));
}

struct Stream {
    Stream() { check(cudaStreamCreate(&value)); }
    ~Stream() { cudaStreamDestroy(value); }
    cudaStream_t value = nullptr;
};

template <class T>
void upload(MACEPolarCuda::Buffer<T>& b, const T* data, size_t n, cudaStream_t stream) {
    b.reserve(n);
    if (n) check(cudaMemcpyAsync(b.p, data, n*sizeof(T), cudaMemcpyHostToDevice, stream));
}

__global__ void scatter(int edges, const int* centers, const int* neighbors,
                        const double* edge_forces, double* forces) {
    const long q = long(blockIdx.x)*blockDim.x + threadIdx.x;
    if (q >= 3L*edges) return;
    const long e = q/3;
    const int k = q%3;
    const double f = edge_forces[q];
    atomicAdd(forces + 3*centers[e] + k, -f);
    atomicAdd(forces + 3*neighbors[e] + k, f);
}

template <realspace::Operation op>
__global__ void realspace_kernel(int n, const double* positions, const realspace::Projection* params,
                                 int widths, const float* q, const float* adj, float* out, double* f) {
    const long thread = long(blockIdx.x)*blockDim.x + threadIdx.x;
    const int i = int(thread/32);
    if (i < n) realspace::evaluate<op>(i, threadIdx.x%32, 32, n, positions, params, widths, q, adj, out, f);
}

__global__ void molecular_dipole(int n, const double* positions, const float* rho, double* out) {
    const int axis = blockIdx.x;
    double sum = 0;
    for (long i = threadIdx.x; i < n; i += blockDim.x)
        sum += atomic_dipole(rho, positions, n, int(i), axis);
    using Reduce = cub::BlockReduce<double, 128>;
    __shared__ Reduce::TempStorage temp;
    const double total = Reduce(temp).Sum(sum);
    if (threadIdx.x == 0) out[axis] = total;
}

struct CUDA : Evaluator {
    CUDA(const std::string& file, double skin, bool mixed)
        : model(file, mixed), geometry(model.r_cut, model.radial_spline_r_min, model.atomic_numbers, skin) {}

    double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                   double charge, double spin, double* forces, double* virial, double* energies,
                   double* dipole) override {
        geometry.update(n, x, numbers, cell, pbc);
        const auto stream = execution.value;
        const size_t edges = geometry.r.size();
        upload(node_types, geometry.node_types.data(), n, stream);
        upload(num_neigh, geometry.num_neigh.data(), n, stream);
        upload(neigh_indices, geometry.neigh_indices.data(), edges, stream);
        upload(neigh_types, geometry.neigh_types.data(), edges, stream);
        upload(centers, geometry.centers.data(), edges, stream);
        upload(xyz, geometry.xyz.data(), 3*edges, stream);
        upload(r, geometry.r.data(), edges, stream);
        upload(positions, x, 3L*n, stream);
        if (!pbc[0] && !device_params.p)
            upload(device_params, params.data(), params.size(), stream);
        realspace::set_hooks(model, !pbc[0], [this, n, stream]<realspace::Operation op>(
                const float* q, const float* adj, float* out, double* f) {
            realspace_kernel<op><<<(n+3L)/4, 128, 0, stream>>>(n, positions.p, device_params.p,
                int(params.size())-1, q, adj, out, f);
            check(cudaGetLastError());
        });
        // The evaluator expects positions/edges on device, but cell metadata on host.
        model.compute_node_energies_forces(n,
            std::span<const int>(node_types.p, n), std::span<const int>(num_neigh.p, n),
            std::span<const int>(neigh_indices.p, edges), std::span<const int>(neigh_types.p, edges),
            std::span<const double>(xyz.p, 3*edges), std::span<const double>(r.p, edges),
            std::span<const double>(positions.p, 3L*n), std::span<const double>(cell, 9),
            charge, spin, stream);
        f.reserve(3L*n);
        check(cudaMemcpyAsync(f.p, model.position_forces.p, 3L*n*sizeof(double), cudaMemcpyDeviceToDevice, stream));
        if (edges) {
            scatter<<<(3*edges + 255)/256, 256, 0, stream>>>(int(edges), centers.p, neigh_indices.p,
                                                           model.node_forces.p, f.p);
            check(cudaGetLastError());
        }
        if (dipole) {
            mu.reserve(3);
            molecular_dipole<<<3, 128, 0, stream>>>(n, positions.p, model.rho.p, mu.p);
            check(cudaGetLastError());
            check(cudaMemcpyAsync(dipole, mu.p, 3*sizeof(double), cudaMemcpyDeviceToHost, stream));
        }
        node_energies.resize(n);
        check(cudaMemcpyAsync(node_energies.data(), model.node_energies.p, n*sizeof(double), cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(forces, f.p, 3L*n*sizeof(double), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        std::copy(model.virial.begin(), model.virial.end(), virial);
        if (energies) std::copy(node_energies.begin(), node_energies.end(), energies);
        return std::accumulate(node_energies.begin(), node_energies.end(), 0.0);
    }

    int neighbor_list_builds() const override { return geometry.builds; }
    Stream execution;
    MACEPolarCuda model;
    Geometry geometry;
    std::vector<realspace::Projection> params = realspace::projections(model);
    MACEPolarCuda::Buffer<realspace::Projection> device_params;
    MACEPolarCuda::Buffer<int> node_types, num_neigh, neigh_indices, neigh_types, centers;
    MACEPolarCuda::Buffer<double> xyz, r, positions, f, mu;
    std::vector<double> node_energies;
};

}  // namespace

std::unique_ptr<Evaluator> cuda(const std::string& file, double skin, bool mixed) {
    return std::make_unique<CUDA>(file, skin, mixed);
}

}  // namespace mace_polar_ase
