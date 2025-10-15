#!/bin/bash
# LinearAlgebra_cuda 环境配置文件
# 用途: 设置库路径，便于在不同环境下使用
# 使用方法: source env_setup.sh

# ============================================================================
# 库路径配置（根据您的实际安装位置修改）
# ============================================================================

# AMGX 路径
export AMGX_DIR="/home/zzy/Plasma/gpu/amgx/install"

# HYPRE 路径（CPU 和 GPU 两个版本）
export HYPRE_CPU_DIR="/home/zzy/Plasma/gpu/hypre_cpu"
export HYPRE_GPU_DIR="/home/zzy/Plasma/gpu/hypre_gpu"

# PETSc 路径
export PETSC_DIR="/home/zzy/Plasma/gpu/petsc-gpu/install"

# yaml-cpp 路径
export YAML_CPP_DIR="/home/zzy/Plasma/gpu/ltpDeps-v2412/extract/yaml-cpp-master"

# ============================================================================
# CUDA 和 MPI 路径（通常不需要修改）
# ============================================================================

# CUDA 路径
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# MPI 路径（如果需要）
# export MPI_HOME="/usr/lib/x86_64-linux-gnu/openmpi"
# export PATH="$MPI_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$MPI_HOME/lib:$LD_LIBRARY_PATH"

# ============================================================================
# 运行时库路径（确保能找到动态库）
# ============================================================================

# AMGX 库
export LD_LIBRARY_PATH="$AMGX_DIR/lib:$LD_LIBRARY_PATH"

# HYPRE 库（如果是动态库）
# export LD_LIBRARY_PATH="$HYPRE_CPU_DIR/lib:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="$HYPRE_GPU_DIR/lib:$LD_LIBRARY_PATH"

# PETSc 库
export LD_LIBRARY_PATH="$PETSC_DIR/lib:$LD_LIBRARY_PATH"

# yaml-cpp 库
# export LD_LIBRARY_PATH="$YAML_CPP_DIR/build:$LD_LIBRARY_PATH"

# ============================================================================
# 环境变量显示
# ============================================================================

echo "================================================"
echo "LinearAlgebra_cuda 环境配置已加载"
echo "================================================"
echo ""
echo "库路径:"
echo "  AMGX_DIR       = $AMGX_DIR"
echo "  HYPRE_CPU_DIR  = $HYPRE_CPU_DIR"
echo "  HYPRE_GPU_DIR  = $HYPRE_GPU_DIR"
echo "  PETSC_DIR      = $PETSC_DIR"
echo "  YAML_CPP_DIR   = $YAML_CPP_DIR"
echo ""
echo "CUDA 路径:"
echo "  CUDA_HOME      = $CUDA_HOME"
echo ""
echo "使用方法:"
echo "  1. 修改本文件中的路径（第 10-20 行）"
echo "  2. source env_setup.sh"
echo "  3. python3 rebuild.py"
echo ""
echo "================================================"

# ============================================================================
# 验证函数（可选）
# ============================================================================

verify_paths() {
    echo "验证库路径..."
    local all_ok=true
    
    # 检查 AMGX
    if [ -f "$AMGX_DIR/lib/libamgxsh.so" ]; then
        echo "  ✅ AMGX: $AMGX_DIR"
    else
        echo "  ❌ AMGX 未找到: $AMGX_DIR/lib/libamgxsh.so"
        all_ok=false
    fi
    
    # 检查 HYPRE CPU
    if [ -f "$HYPRE_CPU_DIR/lib/libHYPRE.a" ]; then
        echo "  ✅ HYPRE CPU: $HYPRE_CPU_DIR"
    else
        echo "  ❌ HYPRE CPU 未找到: $HYPRE_CPU_DIR/lib/libHYPRE.a"
        all_ok=false
    fi
    
    # 检查 HYPRE GPU
    if [ -f "$HYPRE_GPU_DIR/lib/libHYPRE.a" ]; then
        echo "  ✅ HYPRE GPU: $HYPRE_GPU_DIR"
    else
        echo "  ❌ HYPRE GPU 未找到: $HYPRE_GPU_DIR/lib/libHYPRE.a"
        all_ok=false
    fi
    
    # 检查 PETSc
    if [ -f "$PETSC_DIR/lib/libpetsc.so" ]; then
        echo "  ✅ PETSc: $PETSC_DIR"
    else
        echo "  ❌ PETSc 未找到: $PETSC_DIR/lib/libpetsc.so"
        all_ok=false
    fi
    
    # 检查 yaml-cpp
    if [ -f "$YAML_CPP_DIR/build/libyaml-cpp.a" ]; then
        echo "  ✅ yaml-cpp: $YAML_CPP_DIR"
    else
        echo "  ⚠️  yaml-cpp 未找到: $YAML_CPP_DIR/build/libyaml-cpp.a"
        echo "     (可能需要编译: cd $YAML_CPP_DIR/build && cmake .. && make)"
        all_ok=false
    fi
    
    echo ""
    if [ "$all_ok" = true ]; then
        echo "✅ 所有库路径验证通过！"
    else
        echo "❌ 部分库路径有问题，请检查并修改 env_setup.sh"
    fi
    echo ""
}

# 自动验证（注释掉此行可禁用自动验证）
# verify_paths

