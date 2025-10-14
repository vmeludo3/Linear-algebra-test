# PETSc 测试

# 查找 PETSc
set(PETSC_INCLUDE_DIR "${PETSC_DIR}/include")
set(PETSC_LIB_DIR "${PETSC_DIR}/lib")

if(EXISTS ${PETSC_INCLUDE_DIR}/petsc.h)
    message(STATUS "Found PETSc:")
    message(STATUS "  Include: ${PETSC_INCLUDE_DIR}")
    message(STATUS "  Library: ${PETSC_LIB_DIR}")

    include_directories(${PETSC_INCLUDE_DIR})
    link_directories(${PETSC_LIB_DIR})

    # PETSc Poisson 求解器示例
    add_executable(petsc_poisson_solver
        poisson_solver.cpp
    )

    target_link_libraries(petsc_poisson_solver
        benchmark_common
        petsc
        ${MPI_LIBRARIES}
        ${CUDA_LIBRARIES}
    )
else()
    message(WARNING "PETSc not found. Skipping PETSc tests.")
endif()

