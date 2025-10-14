# HYPRE GPU 测试

# 使用 HYPRE GPU 目录
set(HYPRE_GPU_INCLUDE_DIR "${HYPRE_GPU_DIR}/include")
set(HYPRE_GPU_LIBRARY "${HYPRE_GPU_DIR}/lib/libHYPRE.a")

# 查找 HYPRE GPU
find_path(HYPRE_INCLUDE_DIR HYPRE.h
    PATHS ${HYPRE_GPU_INCLUDE_DIR}
    NO_DEFAULT_PATH
)

find_library(HYPRE_LIBRARY HYPRE
    PATHS ${HYPRE_GPU_DIR}/lib
    NO_DEFAULT_PATH
)

if(HYPRE_INCLUDE_DIR AND HYPRE_LIBRARY)
    message(STATUS "Found HYPRE (GPU mode):")
    message(STATUS "  Include: ${HYPRE_INCLUDE_DIR}")
    message(STATUS "  Library: ${HYPRE_LIBRARY}")

    include_directories(${HYPRE_INCLUDE_DIR})
    link_directories(/usr/local/cuda/lib64)

    # HYPRE GPU Poisson 求解器
    add_executable(hypre_gpu_poisson_solver
        poisson_solver.cpp
    )

    target_link_libraries(hypre_gpu_poisson_solver
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
    message(WARNING "HYPRE not found. Skipping HYPRE GPU tests.")
endif()

