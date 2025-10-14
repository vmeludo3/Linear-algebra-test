# AMGX GPU Native Tests CMakeLists.txt

# 设置 AMGX 路径
set(AMGX_INCLUDE_DIR "${AMGX_DIR}/include")
set(AMGX_LIBRARY "${AMGX_DIR}/lib/libamgxsh.so")

# 创建可执行文件
add_executable(amgx_gpu_native_poisson_solver poisson_solver.cu)

# 设置为 CUDA 可执行文件
set_target_properties(amgx_gpu_native_poisson_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 包含目录
target_include_directories(amgx_gpu_native_poisson_solver PRIVATE
    ${AMGX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${YAML_CPP_INCLUDE_DIR}
)

# 链接库
target_link_libraries(amgx_gpu_native_poisson_solver
    ${AMGX_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    benchmark_common
    ${YAML_CPP_LIBRARY}
    gomp  # OpenMP support for AMGX
)

# 设置 CUDA 架构
set_property(TARGET amgx_gpu_native_poisson_solver 
             PROPERTY CUDA_ARCHITECTURES 89)

# 设置输出目录
set_target_properties(amgx_gpu_native_poisson_solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/amgx_gpu_native_tests"
)

