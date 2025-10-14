#!/bin/bash

# 运行所有性能测试

set -e

echo "=========================================="
echo "Running LinearAlgebra_cuda Benchmarks"
echo "=========================================="
echo ""

# 确保 results 目录存在
mkdir -p results

cd build

# 运行 AMGX 测试
if [ -f "amgx_tests/amgx_poisson_solver" ]; then
    echo "Running AMGX tests..."
    ./amgx_tests/amgx_poisson_solver
    echo ""
fi

# 运行 HYPRE 测试
if [ -f "hypre_tests/hypre_poisson_solver" ]; then
    echo "Running HYPRE tests..."
    mpirun -np 1 ./hypre_tests/hypre_poisson_solver
    echo ""
fi

# 运行 PETSc 测试
if [ -f "petsc_tests/petsc_poisson_solver" ]; then
    echo "Running PETSc tests..."
    mpirun -np 1 ./petsc_tests/petsc_poisson_solver
    echo ""
fi

echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="
echo ""
echo "Results saved in: results/"
ls -lh ../results/

