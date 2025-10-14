# HYPRE 测试

# 查找 HYPRE
find_path(HYPRE_INCLUDE_DIR HYPRE.h
    PATHS ${HYPRE_DIR}/include
    NO_DEFAULT_PATH
)

find_library(HYPRE_LIBRARY HYPRE
    PATHS ${HYPRE_DIR}/lib
    NO_DEFAULT_PATH
)

if(HYPRE_INCLUDE_DIR AND HYPRE_LIBRARY)
    message(STATUS "Found HYPRE:")
    message(STATUS "  Include: ${HYPRE_INCLUDE_DIR}")
    message(STATUS "  Library: ${HYPRE_LIBRARY}")

    include_directories(${HYPRE_INCLUDE_DIR})
    link_directories(/usr/local/cuda/lib64)

    # HYPRE Poisson 求解器示例
    add_executable(hypre_poisson_solver
        poisson_solver.cpp
    )

    target_link_libraries(hypre_poisson_solver
        benchmark_common
        ${HYPRE_LIBRARY}
        ${MPI_LIBRARIES}
        ${CUDA_LIBRARIES}
        -L/usr/local/cuda/lib64
        -lcusparse
        -lcublas
        -lcurand
        -lcusolver
    )
else()
    message(WARNING "HYPRE not found. Skipping HYPRE tests.")
endif()

