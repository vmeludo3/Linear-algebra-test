# PETSc GPU Native Tests CMakeLists.txt

# PETSc 配置
set(PETSC_INCLUDE_DIR "${PETSC_DIR}/include")
set(PETSC_LIBRARY "${PETSC_DIR}/lib/libpetsc.so")

# 创建可执行文件
add_executable(petsc_gpu_native_poisson_solver poisson_solver.cu)

# 设置为 CUDA 可执行文件
set_target_properties(petsc_gpu_native_poisson_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 包含目录
target_include_directories(petsc_gpu_native_poisson_solver PRIVATE
    ${PETSC_INCLUDE_DIR}
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${YAML_CPP_INCLUDE_DIR}
)

# 链接库
target_link_libraries(petsc_gpu_native_poisson_solver
    ${PETSC_LIBRARY}
    ${MPI_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    benchmark_common
    ${YAML_CPP_LIBRARY}
)

# 设置 CUDA 架构
set_property(TARGET petsc_gpu_native_poisson_solver 
             PROPERTY CUDA_ARCHITECTURES 89)

# 设置输出目录
set_target_properties(petsc_gpu_native_poisson_solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/petsc_gpu_native_tests"
)

