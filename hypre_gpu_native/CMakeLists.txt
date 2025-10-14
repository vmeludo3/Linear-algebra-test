# HYPRE GPU Native Tests CMakeLists.txt

# 设置 HYPRE 路径
set(HYPRE_INCLUDE_DIR "${HYPRE_DIR}/include")
set(HYPRE_LIBRARY "${HYPRE_DIR}/lib/libHYPRE.a")

# 创建可执行文件
add_executable(hypre_gpu_native_poisson_solver_working poisson_solver_v3.cpp)

# Working版本设置 (.cpp 文件，避免CUDA编译器冲突)
target_include_directories(hypre_gpu_native_poisson_solver_working PRIVATE
    ${HYPRE_INCLUDE_DIR}
    ${MPI_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(hypre_gpu_native_poisson_solver_working
    ${HYPRE_LIBRARY}
    ${MPI_LIBRARIES}
    -L/usr/local/cuda/lib64
    -lcusparse
    -lcublas
    -lcurand
    -lcusolver
    benchmark_common
    ${YAML_CPP_LIBRARY}
)

set_target_properties(hypre_gpu_native_poisson_solver_working PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/hypre_gpu_native_tests"
)

