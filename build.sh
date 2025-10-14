#!/bin/bash

# LinearAlgebra_cuda 构建脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Building LinearAlgebra_cuda Benchmarks"
echo "=========================================="
echo ""

# 创建 build 目录
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# 清理旧的构建文件
echo "Cleaning old build files..."
rm -rf *

# 运行 CMake
echo ""
echo "Running CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DHYPRE_CPU_DIR=/home/zzy/Plasma/gpu/hypre_cpu \
    -DHYPRE_GPU_DIR=/home/zzy/Plasma/gpu/hypre_gpu \
    -DPETSC_DIR=/home/zzy/Plasma/gpu/petsc-gpu/install \
    -DAMGX_DIR=/home/zzy/Plasma/gpu/amgx/install

# 编译
echo ""
echo "Compiling..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Available executables:"
if [ -f "amgx_tests/amgx_poisson_solver" ]; then
    echo "  - amgx_tests/amgx_poisson_solver"
fi
if [ -f "hypre_tests/hypre_poisson_solver" ]; then
    echo "  - hypre_tests/hypre_poisson_solver"
fi
if [ -f "petsc_tests/petsc_poisson_solver" ]; then
    echo "  - petsc_tests/petsc_poisson_solver"
fi
echo ""
echo "To run tests:"
echo "  cd build"
echo "  ./amgx_tests/amgx_poisson_solver"
echo ""

