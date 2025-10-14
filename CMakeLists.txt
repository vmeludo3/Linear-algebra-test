cmake_minimum_required(VERSION 3.18)
project(LinearAlgebra_CUDA_Benchmarks LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# 查找依赖
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

# 设置库路径
set(HYPRE_CPU_DIR "/home/zzy/Plasma/gpu/hypre_cpu" CACHE PATH "HYPRE CPU installation directory")
set(HYPRE_GPU_DIR "/home/zzy/Plasma/gpu/hypre_gpu" CACHE PATH "HYPRE GPU installation directory")
set(PETSC_DIR "/home/zzy/Plasma/gpu/petsc-gpu/install" CACHE PATH "PETSc directory")
set(AMGX_DIR "/home/zzy/Plasma/gpu/amgx/install" CACHE PATH "AMGX installation directory")
set(YAML_CPP_DIR "/home/zzy/Plasma/gpu/ltpDeps-v2412/extract/yaml-cpp-master" CACHE PATH "yaml-cpp directory")

# yaml-cpp 支持
set(YAML_CPP_INCLUDE_DIR "${YAML_CPP_DIR}/include")
set(YAML_CPP_LIBRARY "${YAML_CPP_DIR}/build/libyaml-cpp.a")

# 通用包含目录和库
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CUDA_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
    ${YAML_CPP_INCLUDE_DIR}
)

# 添加通用工具库
add_subdirectory(common)

# 选项：启用/禁用特定库的测试
option(BUILD_HYPRE_TESTS "Build HYPRE CPU tests" ON)
option(BUILD_HYPRE_GPU_TESTS "Build HYPRE GPU tests" ON)
option(BUILD_PETSC_TESTS "Build PETSc CPU tests" ON)
option(BUILD_PETSC_GPU_TESTS "Build PETSc GPU tests" ON)
option(BUILD_AMGX_TESTS "Build AMGX tests" ON)
option(BUILD_AMGX_GPU_NATIVE_TESTS "Build AMGX GPU Native tests" ON)
option(BUILD_HYPRE_GPU_NATIVE_TESTS "Build HYPRE GPU Native tests" ON)
option(BUILD_PETSC_GPU_NATIVE_TESTS "Build PETSc GPU Native tests" ON)

# HYPRE CPU 测试
if(BUILD_HYPRE_TESTS)
    if(EXISTS ${HYPRE_CPU_DIR})
        set(HYPRE_DIR ${HYPRE_CPU_DIR})
        add_subdirectory(hypre_tests)
        message(STATUS "HYPRE CPU tests enabled: ${HYPRE_CPU_DIR}")
    else()
        message(WARNING "HYPRE CPU directory not found: ${HYPRE_CPU_DIR}")
    endif()
endif()

# HYPRE GPU 测试
if(BUILD_HYPRE_GPU_TESTS)
    if(EXISTS ${HYPRE_GPU_DIR})
        set(HYPRE_DIR ${HYPRE_GPU_DIR})
        add_subdirectory(hypre_gpu_tests)
        message(STATUS "HYPRE GPU tests enabled: ${HYPRE_GPU_DIR}")
    else()
        message(WARNING "HYPRE GPU directory not found: ${HYPRE_GPU_DIR}")
    endif()
endif()

# PETSc CPU 测试
if(BUILD_PETSC_TESTS)
    if(EXISTS ${PETSC_DIR})
        add_subdirectory(petsc_tests)
        message(STATUS "PETSc CPU tests enabled: ${PETSC_DIR}")
    else()
        message(WARNING "PETSc directory not found: ${PETSC_DIR}")
    endif()
endif()

# PETSc GPU 测试
if(BUILD_PETSC_GPU_TESTS)
    if(EXISTS ${PETSC_DIR})
        add_subdirectory(petsc_gpu_tests)
        message(STATUS "PETSc GPU tests enabled: ${PETSC_DIR}")
    else()
        message(WARNING "PETSc directory not found: ${PETSC_DIR}")
    endif()
endif()

# AMGX 测试
if(BUILD_AMGX_TESTS)
    if(EXISTS ${AMGX_DIR})
        add_subdirectory(amgx_tests)
        message(STATUS "AMGX tests enabled: ${AMGX_DIR}")
    else()
        message(WARNING "AMGX directory not found: ${AMGX_DIR}")
    endif()
endif()

# AMGX GPU Native 测试
if(BUILD_AMGX_GPU_NATIVE_TESTS)
    if(EXISTS ${AMGX_DIR})
        add_subdirectory(amgx_gpu_native)
        message(STATUS "AMGX GPU Native tests enabled: ${AMGX_DIR}")
    else()
        message(WARNING "AMGX directory not found: ${AMGX_DIR}")
    endif()
endif()

# HYPRE GPU Native 测试
if(BUILD_HYPRE_GPU_NATIVE_TESTS)
    if(EXISTS ${HYPRE_GPU_DIR})
        set(HYPRE_DIR ${HYPRE_GPU_DIR})
        add_subdirectory(hypre_gpu_native)
        message(STATUS "HYPRE GPU Native tests enabled: ${HYPRE_GPU_DIR}")
    else()
        message(WARNING "HYPRE GPU directory not found: ${HYPRE_GPU_DIR}")
    endif()
endif()

# PETSc GPU Native 测试
if(BUILD_PETSC_GPU_NATIVE_TESTS)
    if(EXISTS ${PETSC_DIR})
        add_subdirectory(petsc_gpu_native)
        message(STATUS "PETSc GPU Native tests enabled: ${PETSC_DIR}")
    else()
        message(WARNING "PETSc directory not found: ${PETSC_DIR}")
    endif()
endif()

# 打印配置信息
message(STATUS "===========================================")
message(STATUS "Linear Algebra CUDA Benchmarks Configuration")
message(STATUS "===========================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "MPI Found: ${MPI_FOUND}")
message(STATUS "===========================================")

