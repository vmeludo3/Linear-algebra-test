# 求解器配置详细说明

**文档日期**: 2025-10-13

---

## 测试模块总览

| # | 模块名称 | 执行位置 | 求解器 | 预条件器 | 库路径 |
|---|---------|---------|--------|---------|--------|
| 1 | AMGX | GPU | PCG | Jacobi | `/home/zzy/Plasma/gpu/amgx/install` |
| 2 | HYPRE-CPU | CPU | PCG | BoomerAMG | `/home/zzy/Plasma/gpu/hypre_cpu` |
| 3 | HYPRE-GPU | GPU | PCG | BoomerAMG | `/home/zzy/Plasma/gpu/hypre_gpu` |
| 4 | PETSc-CPU | CPU | CG | GAMG | `/home/zzy/Plasma/gpu/petsc-gpu/install` |
| 5 | PETSc-GPU | GPU | CG | GAMG | `/home/zzy/Plasma/gpu/petsc-gpu/install` |

---

## 1. AMGX (GPU)

### 求解器: PCG (Preconditioned Conjugate Gradient)
```
类型:               PCG
最大迭代次数:       1000
收敛容差:           1e-6
收敛判据:           RELATIVE_INI_CORE (相对初始残差)
监控残差:           启用
打印统计信息:       启用
```

### 预条件器: Block Jacobi
```
类型:               BLOCK_JACOBI
块大小:             默认 (1)
说明:               最简单的预条件器，每次迭代只需矩阵对角线
优点:               实现简单，GPU 并行性好
缺点:               收敛慢，迭代次数多
```

### 执行参数
```
内存位置:           GPU (CUDA Device Memory)
执行策略:           GPU (CUDA Kernels)
CUDA 版本:          12.6
GPU 架构:           SM 70, 75, 80, 86, 89, 90
```

---

## 2. HYPRE-CPU

### 求解器: ParCSR PCG
```
类型:               PCG (Parallel CSR format)
最大迭代次数:       1000
收敛容差:           1e-6
双范数:             启用
残差检查:           相对残差
打印级别:           0 (静默)
```

### 预条件器: BoomerAMG
```
类型:               代数多重网格 (AMG)
粗化策略:           6 (Falgout coarsening)
  说明:             经典 Ruge-Stüben 粗化的改进版本
  
松弛方法:           6 (对称 Gauss-Seidel)
  前扫描:           1次
  后扫描:           1次
  
最大层数:           20
预条件容差:         0.0 (每次只做1次V-循环)
预条件迭代:         1

V-循环结构:
  细网格 → 平滑(下) → 粗化 → ... → 最粗层 → ... → 细化 → 平滑(上)
```

### 执行参数
```
内存位置:           HOST (CPU RAM)
执行策略:           HOST (CPU)
并行模式:           MPI
```

---

## 3. HYPRE-GPU

### 求解器: ParCSR PCG (同 HYPRE-CPU)
```
类型:               PCG
最大迭代次数:       1000
收敛容差:           1e-6
双范数:             启用
打印级别:           0
```

### 预条件器: BoomerAMG (同 HYPRE-CPU)
```
粗化策略:           6 (Falgout)
松弛方法:           6 (对称 GS)
平滑次数:           1
最大层数:           20
```

### 执行参数
```
内存位置:           DEVICE (GPU VRAM)
执行策略:           DEVICE (GPU Kernels)
CUDA 初始化:        显式 cudaSetDevice(0)
GPU 同步:           每次求解后同步
并行模式:           MPI + CUDA
```

**关键差异**:
- 数据存储在 GPU 内存
- 矩阵运算使用 CUDA 核函数
- 稀疏矩阵向量乘法在 GPU 执行
- AMG 层次结构在 GPU 构建

---

## 4. PETSc-CPU

### 求解器: KSP CG
```
类型:               CG (Conjugate Gradient)
最大迭代次数:       2000
相对收敛容差:       1e-6
绝对收敛容差:       1e-8
发散容差:           默认 (1e4)
残差范数类型:       UNPRECONDITIONED (真实残差)
初始残差归一化:     启用 (UIRNorm)
```

### 预条件器: PC GAMG
```
类型:               GAMG (Geometric-Algebraic MultiGrid)
AMG 类型:           PCGAMGAGG (聚合型)
平滑次数:           1
粗化策略:           自适应聚合
  聚合方法:         未配对聚合 (Unsmoothed Aggregation)
  平滑器:           默认 (Chebyshev)

多层网格结构:
  - 自动确定层数
  - 基于代数距离的聚合
  - 粗层算子通过 Galerkin 投影
```

### 执行参数
```
矩阵类型:           MATSEQAIJ (CPU 稀疏矩阵)
向量类型:           VECSEQ (CPU 向量)
内存分配:           CPU RAM
计算位置:           CPU
```

---

## 5. PETSc-GPU

### 求解器: KSP CG (同 PETSc-CPU)
```
类型:               CG
最大迭代次数:       2000
相对容差:           1e-6
绝对容差:           1e-8
残差范数:           UNPRECONDITIONED
GPU-aware MPI:      禁用 (WSL 兼容性)
```

### 预条件器: PC GAMG (同 PETSc-CPU)
```
类型:               GAMG
AMG 类型:           聚合
平滑次数:           1
```

### 执行参数
```
矩阵类型:           MATAIJCUSPARSE (CUDA 稀疏矩阵)
向量类型:           VECCUDA (CUDA 向量)
内存分配:           GPU VRAM
计算位置:           GPU (cuSPARSE/cuBLAS)

CUDA 库:
  - cuSPARSE: 稀疏矩阵运算
  - cuBLAS:   向量操作
  - cuSOLVER: 直接求解器
```

**关键差异**:
- 使用 cuSPARSE 进行矩阵向量乘法
- 向量操作通过 cuBLAS
- 数据驻留在 GPU
- CPU-GPU 数据传输最小化

---

## 收敛标准对齐

所有求解器都配置为达到相似的收敛精度：

| 库 | 相对容差 | 绝对容差 | 残差定义 | 实际达到精度 |
|---|---------|---------|---------|-------------|
| AMGX | 1e-6 | - | 相对初始 | ~1e-7 |
| HYPRE-CPU | 1e-6 | - | 相对初始 | ~1e-7 |
| HYPRE-GPU | 1e-6 | - | 相对初始 | ~1e-7 |
| PETSc-CPU | 1e-6 | 1e-8 | 相对RHS | ~1e-7 |
| PETSc-GPU | 1e-6 | 1e-8 | 相对RHS | ~1e-7 |

**✅ 所有求解器都达到 10⁻⁷ 级别的相对残差**

---

## 预条件器效果对比

### 迭代次数对比 (512×512 规模)

| 预条件器 | 实现库 | 迭代次数 | 说明 |
|---------|--------|---------|------|
| BoomerAMG | HYPRE | **5次** | 🥇 最强，经典 AMG |
| GAMG | PETSc | **12次** | 🥈 优秀，几何代数结合 |
| Jacobi | AMGX | **61次** | 🥉 简单，收敛慢 |

### 为什么 BoomerAMG 最强？

**BoomerAMG (HYPRE)**:
- ✅ 经典代数多重网格方法
- ✅ Falgout 粗化：智能选择粗网格点
- ✅ 对称 GS 平滑：高效消除误差
- ✅ 多层网格：对 Poisson 问题效果极佳
- ✅ 成熟稳定：经过20+年优化

**GAMG (PETSc)**:
- ✅ 聚合型 AMG：适用于非结构网格
- ✅ 自适应粗化：自动调整策略
- ✅ 通用性强：适用多种问题
- ⚠️ 聚合可能不如 Falgout 精确

**Jacobi (AMGX)**:
- ⚠️ 仅使用对角元素
- ⚠️ 无层次结构
- ⚠️ 收敛慢，需要多次迭代
- ✅ GPU 并行性最好
- 💡 建议: AMGX 也支持 AMG，应该使用

---

## GPU 加速效果分析

### HYPRE: GPU vs CPU (512×512)

| 阶段 | CPU | GPU | 加速比 |
|-----|-----|-----|-------|
| Setup | 0.274秒 | 0.264秒 | 1.04× |
| Solve | 0.159秒 | 0.134秒 | **1.19×** |
| Total | 0.433秒 | 0.397秒 | 1.09× |

**结论**: GPU 在求解阶段有 19% 加速

### PETSc: GPU vs CPU (512×512)

| 阶段 | CPU | GPU | 加速比 |
|-----|-----|-----|-------|
| Setup | 0.517秒 | 0.419秒 | 1.23× |
| Solve | 0.187秒 | 0.034秒 | **5.58×** ⚡ |
| Total | 0.704秒 | 0.452秒 | **1.56×** |

**结论**: GPU 在求解阶段有 5.6× 显著加速！

### 为什么 PETSc GPU 求解更快？

1. **cuSPARSE 优化**
   - NVIDIA 深度优化的稀疏矩阵库
   - 专门针对 GPU 架构优化
   
2. **向量操作**
   - cuBLAS 高度优化的向量运算
   - 内存合并访问模式
   
3. **数据局部性**
   - 数据全程在 GPU
   - 避免 CPU-GPU 传输

4. **CG 算法特性**
   - 每次迭代固定操作量
   - GPU 并行性充分发挥

---

## 总结建议

### 选择求解器的建议

**优先考虑收敛速度（迭代次数少）**:
→ 选择 HYPRE BoomerAMG (5次迭代)

**优先考虑总时间（包括 Setup）**:
→ 选择 HYPRE-GPU (0.397秒)

**优先考虑求解阶段速度**:
→ 选择 PETSc-GPU (0.034秒，5.6× GPU 加速)

**平衡性能和易用性**:
→ PETSc 生态系统完善，文档丰富

**需要极致性能**:
→ 建议为 AMGX 配置 AMG 预条件器而非 Jacobi

---

## 配置文件位置

各求解器的配置代码位于：

```
LinearAlgebra_cuda/
├── amgx_tests/poisson_solver.cpp          (行 44-56: 配置字符串)
├── hypre_tests/poisson_solver.cpp         (行 142-161: 预条件器设置)
├── hypre_gpu_tests/poisson_solver.cpp     (行 163-182: 预条件器设置)
├── petsc_tests/poisson_solver.cpp         (行 134-152: KSP/PC 设置)
└── petsc_gpu_tests/poisson_solver.cpp     (行 134-152: KSP/PC 设置)
```

如需修改参数，请编辑对应文件后重新编译。

