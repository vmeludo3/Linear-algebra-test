# 测试结果归档 - 2025-10-14

**测试日期**: 2025-10-14  
**测试环境**: WSL2 + CUDA 12.6 + RTX 4060 Ti (8GB)

---

## 📚 相关文档

对应的分析文档位于: `../../archived_docs/2025-10-14/`

| 文档 | 描述 |
|------|------|
| **[PERFORMANCE_COMPARISON.md](../../archived_docs/2025-10-14/PERFORMANCE_COMPARISON.md)** | 完整性能对比报告 ⭐ |
| **[SOLVER_CONFIGURATIONS.md](../../archived_docs/2025-10-14/SOLVER_CONFIGURATIONS.md)** | 求解器配置详解 ⭐ |
| **[FINAL_SUMMARY.md](../../archived_docs/2025-10-14/FINAL_SUMMARY.md)** | 项目最终总结 |

---

## 测试配置

### 测试的库和模块

| 模块 | 库 | 版本 | 模式 | 求解器 | 预条件器 | 状态 |
|------|---|------|------|--------|----------|------|
| 1 | AMGX | v2.4.0 | GPU | PCG | AMG | ✅ |
| 2 | HYPRE | 2.31.0 | CPU | PCG | BoomerAMG | ✅ |
| 3 | HYPRE | 2.31.0 | GPU | PCG | BoomerAMG | ✅ |
| 4 | PETSc | 3.22.2 | CPU | CG | GAMG | ✅ |
| 5 | PETSc | 3.22.2 | GPU | CG | GAMG | ✅ |

### 测试问题
- **问题类型**: 2D Poisson 方程 (-∇²u = f)
- **离散方法**: 5点有限差分
- **网格规模**: 64×64, 128×128, 256×256, 512×512
- **收敛标准**: 相对残差 < 1e-6

---

## 结果文件

| 文件名 | 描述 | 数据点数 |
|-------|------|---------|
| `amgx_results.csv` | AMGX GPU 测试结果 | 4 |
| `amgx_results.json` | AMGX GPU 测试结果 (JSON) | 4 |
| `hypre_results.csv` | HYPRE CPU 测试结果 | 4 |
| `hypre_results.json` | HYPRE CPU 测试结果 (JSON) | 4 |
| `hypre_gpu_results.csv` | HYPRE GPU 测试结果 | 4 |
| `hypre_gpu_results.json` | HYPRE GPU 测试结果 (JSON) | 4 |
| `petsc_results.csv` | PETSc CPU 测试结果 | 4 |
| `petsc_results.json` | PETSc CPU 测试结果 (JSON) | 4 |
| `petsc_gpu_results.csv` | PETSc GPU 测试结果 | 4 |
| `petsc_gpu_results.json` | PETSc GPU 测试结果 (JSON) | 4 |

**总数据点**: 20个测试用例 (5个模块 × 4个规模)

---

## 性能摘要 (512×512 规模)

| 库 | 求解器 | 预条件器 | Total (s) | Solve (s) | 迭代数 | 相对残差 |
|---|--------|----------|-----------|-----------|--------|----------|
| AMGX-GPU | PCG | AMG | 0.147 🥇 | 0.112 | 65 | 8.73e-07 |
| HYPRE-GPU | PCG | BoomerAMG | 0.397 | 0.134 | 5 🥇 | 9.40e-07 |
| HYPRE-CPU | PCG | BoomerAMG | 0.433 | 0.159 | 5 | 9.40e-07 |
| PETSc-GPU | CG | GAMG | 0.452 | 0.034 🥇 | 12 | 7.37e-07 |
| PETSc-CPU | CG | GAMG | 0.704 | 0.187 | 12 | 7.37e-07 |

**关键发现**:
- 🥇 **总时间最快**: AMGX-GPU (AMG) - 0.147秒
- ⚡ **求解最快**: PETSc-GPU - 0.034秒 (GPU加速5.6×)
- 🎯 **迭代最少**: HYPRE BoomerAMG - 5次

---

## 关键改进

### AMGX 预条件器更新
- **之前**: Jacobi (61次迭代, 0.639秒)
- **现在**: AMG (65次迭代, 0.147秒)
- **改进**: 求解时间快 **4.3倍**

### PETSc GPU 问题解决
- **问题**: CUDA 内核镜像不匹配
- **解决**: 重新编译 PETSc 只针对 SM 89 架构
- **结果**: 成功运行，求解阶段 GPU 加速 5.6×

---

## 技术备注

1. **HYPRE GPU 初始化**
   - 需要显式 `cudaSetDevice(0)` 和 `cudaFree(0)`
   - 然后设置 `HYPRE_MEMORY_DEVICE` 和 `HYPRE_EXEC_DEVICE`

2. **PETSc GPU 配置**
   - 使用 `MATAIJCUSPARSE` 和 `VECCUDA`
   - 禁用 GPU-aware MPI (`-use_gpu_aware_mpi 0`)

3. **精度对齐**
   - 所有库都达到 ~1e-7 相对残差
   - PETSc 手动计算相对残差以对齐标准

---

**归档日期**: 2025-10-14  
**测试完成时间**: 约 21:50

