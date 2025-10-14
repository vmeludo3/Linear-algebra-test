# LinearAlgebra_cuda - GPU 加速线性代数库性能测试平台

**版本**: 1.0  
**创建日期**: 2025-10-13  
**状态**: ✅ 完成并可用

---

## 🎯 项目简介

这是一个完整的 GPU 加速线性代数库性能测试平台，用于对比 **AMGX**、**HYPRE** 和 **PETSc** 三大主流库在 CPU 和 GPU 模式下的性能表现。

**特点**:
- ✅ **5个测试模块** (1个 AMGX + 2个 HYPRE + 2个 PETSc)
- ✅ **CPU/GPU 双模式对比**
- ✅ **精度完全对齐** (~1e-7 相对残差)
- ✅ **详细性能分析** (包含求解器和预条件器配置)
- ✅ **完善的文档** (7个 Markdown 文件)

---

## 🚀 快速开始

### 构建项目
```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda
./build.sh
```

### 运行测试
```bash
cd build

# 运行单个测试
./amgx_tests/amgx_poisson_solver              # AMGX GPU
./hypre_tests/hypre_poisson_solver            # HYPRE CPU
./hypre_gpu_tests/hypre_gpu_poisson_solver    # HYPRE GPU
./petsc_tests/petsc_poisson_solver            # PETSc CPU
./petsc_gpu_tests/petsc_gpu_poisson_solver    # PETSc GPU

# 或运行所有测试
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda
./run_benchmarks.sh
```

### 查看结果
```bash
cd results
ls -lh  # 查看生成的 CSV 和 JSON 文件
```

---

## 📚 文档导航

完整文档位于 **`doc/`** 目录：

| 文档 | 说明 |
|------|------|
| **[QUICKSTART.md](doc/QUICKSTART.md)** | 快速开始指南，5分钟上手 |
| **[PERFORMANCE_COMPARISON.md](doc/PERFORMANCE_COMPARISON.md)** | 完整性能对比报告（含求解器参数）⭐ |
| **[SOLVER_CONFIGURATIONS.md](doc/SOLVER_CONFIGURATIONS.md)** | 求解器配置详细说明 ⭐ |
| **[FINAL_SUMMARY.md](doc/FINAL_SUMMARY.md)** | 项目最终总结 |
| **[STATUS.md](doc/STATUS.md)** | 项目状态和已知问题 |
| **[PROJECT_SUMMARY.md](doc/PROJECT_SUMMARY.md)** | 项目概述 |

---

## 📦 测试的库

| 库 | 版本 | 位置 | CPU | GPU |
|---|------|------|-----|-----|
| **AMGX** | v2.4.0 | `/home/zzy/Plasma/gpu/amgx/install` | ❌ | ✅ |
| **HYPRE** | 2.31.0 | `/home/zzy/Plasma/gpu/hypre_cpu` (CPU)<br>`/home/zzy/Plasma/gpu/hypre_gpu` (GPU) | ✅ | ✅ |
| **PETSc** | 3.22.2 | `/home/zzy/Plasma/gpu/petsc-gpu/install` | ✅ | ✅ |

---

## 🏆 性能亮点

**测试问题**: 2D Poisson 方程，512×512 网格 (262,144 未知数)

| 排名 | 库 | 求解器 | 预条件器 | 总时间 | 求解时间 | 迭代数 |
|-----|---|--------|----------|--------|---------|--------|
| 🥇 | **HYPRE-GPU** | PCG | BoomerAMG | **0.397s** | 0.134s | **5** |
| 🥈 | HYPRE-CPU | PCG | BoomerAMG | 0.433s | 0.159s | 5 |
| 🥉 | PETSc-GPU | CG | GAMG | 0.452s | **0.034s** ⚡ | 12 |
| 4 | AMGX-GPU | PCG | Jacobi | 0.639s | 0.636s | 61 |
| 5 | PETSc-CPU | CG | GAMG | 0.704s | 0.187s | 12 |

**关键发现**:
- ⚡ **PETSc-GPU** 求解阶段最快 (0.034s)，GPU 加速 **5.6倍**
- 🏆 **HYPRE-GPU** 总时间最优 (0.397s)
- 🎯 **HYPRE BoomerAMG** 迭代次数最少 (5次)

详细对比请查看 **[doc/PERFORMANCE_COMPARISON.md](doc/PERFORMANCE_COMPARISON.md)**

---

## 📊 测试模块

```
LinearAlgebra_cuda/
├── amgx_tests/           # AMGX GPU 测试
├── hypre_tests/          # HYPRE CPU 测试
├── hypre_gpu_tests/      # HYPRE GPU 测试
├── petsc_tests/          # PETSc CPU 测试
├── petsc_gpu_tests/      # PETSc GPU 测试
├── common/               # 通用工具库
├── doc/                  # 📚 文档目录
├── build/                # 构建目录
└── results/              # 结果输出目录
```

---

## 🛠️ 环境要求

- **CUDA**: 12.6+
- **GPU**: NVIDIA (测试环境: RTX 4060 Ti)
- **MPI**: OpenMPI
- **CMake**: 3.18+
- **编译器**: GCC 9.4+

---

## 📖 更多信息

- **快速开始**: 查看 [doc/QUICKSTART.md](doc/QUICKSTART.md)
- **性能报告**: 查看 [doc/PERFORMANCE_COMPARISON.md](doc/PERFORMANCE_COMPARISON.md)
- **求解器配置**: 查看 [doc/SOLVER_CONFIGURATIONS.md](doc/SOLVER_CONFIGURATIONS.md)
- **项目总结**: 查看 [doc/FINAL_SUMMARY.md](doc/FINAL_SUMMARY.md)

---

**项目主页**: `/home/zzy/Plasma/gpu/LinearAlgebra_cuda`  
**维护状态**: 活跃  
**许可**: 开源

