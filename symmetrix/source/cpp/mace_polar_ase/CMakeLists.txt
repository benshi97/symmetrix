pybind11_add_module(mace_polar_ase_bindings MODULE bindings.cpp cpu.cpp)
set_target_properties(mace_polar_ase_bindings PROPERTIES OUTPUT_NAME _mace_polar)
target_link_libraries(mace_polar_ase_bindings PRIVATE symmetrix symmetrix_ase_neighbors)
if(NOT APPLE)
  target_link_options(mace_polar_ase_bindings PRIVATE -Wl,--exclude-libs,ALL)
endif()

# libsymmetrix owns the model implementation; this module adds only the ASE adapter.
if(SYMMETRIX_KOKKOS AND Kokkos_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(mace_polar_ase_bindings PRIVATE cuda.cu)
  target_compile_definitions(mace_polar_ase_bindings PRIVATE MACE_POLAR_ASE_CUDA)
  target_link_libraries(mace_polar_ase_bindings PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(mace_polar_ase_bindings PROPERTIES CUDA_STANDARD 20
                        CUDA_ARCHITECTURES "${SYMMETRIX_CUDA_ARCHITECTURES}")
  set_source_files_properties(cuda.cu PROPERTIES COMPILE_OPTIONS --expt-relaxed-constexpr)
endif()

install(TARGETS mace_polar_ase_bindings DESTINATION symmetrix)
