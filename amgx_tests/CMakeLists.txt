# AMGX 测试

# 设置 AMGX 路径
set(AMGX_INCLUDE_DIR "${AMGX_DIR}/include")
set(AMGX_LIB_DIR "${AMGX_DIR}/lib")

include_directories(${AMGX_INCLUDE_DIR})
link_directories(${AMGX_LIB_DIR})

# AMGX Poisson 求解器示例
add_executable(amgx_poisson_solver
    poisson_solver.cpp
)

target_link_libraries(amgx_poisson_solver
    benchmark_common
    amgxsh
    ${MPI_LIBRARIES}
    ${CUDA_LIBRARIES}
    gomp
)

message(STATUS "AMGX tests configured")
message(STATUS "  Include: ${AMGX_INCLUDE_DIR}")
message(STATUS "  Library: ${AMGX_LIB_DIR}")

