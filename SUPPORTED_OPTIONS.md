# 求解器和预条件器支持清单

**更新日期**: 2025-10-14  
**版本**: 1.0

---

## 📊 当前实现状态

| 库 | 模式 | 支持的求解器 | 支持的预条件器 | YAML配置 |
|---|------|------------|--------------|---------|
| AMGX | GPU | PCG | AMG, JACOBI | ✅ |
| HYPRE | CPU | PCG | BOOMERAMG, JACOBI, NONE | ✅ |
| HYPRE | GPU | PCG | BOOMERAMG, JACOBI, NONE | ✅ |
| PETSc | CPU | CG | GAMG, JACOBI, NONE | ✅ |
| PETSc | GPU | CG | GAMG, JACOBI, NONE | ✅ |

---

## 1. AMGX (GPU)

### 当前实现的求解器
- ✅ **PCG** (Preconditioned Conjugate Gradient)

### 当前实现的预条件器

| 预条件器 | YAML配置 | 状态 | 说明 |
|---------|----------|------|------|
| **AMG** | `AMG` | ✅ 已实现 | 代数多重网格，推荐使用 |
| **Jacobi** | `JACOBI` | ✅ 已实现 | 块Jacobi，简单但收敛慢 |

### AMGX 库支持但未实现的选项

**求解器**:
- GMRES (广义最小残差)
- BiCGSTAB (双共轭梯度稳定法)
- CG (普通共轭梯度)
- FGMRES (灵活GMRES)
- IDR (诱导维度缩减)

**预条件器**:
- AMG 的其他配置:
  - `selector`: SIZE_2 (当前), HMIS, PMIS, MULTI_PAIRWISE
  - `algorithm`: AGGREGATION (当前), CLASSICAL
  - `smoother`: BLOCK_JACOBI (当前), MULTICOLOR_GS, MULTICOLOR_DILU
  - `cycle`: V (当前), W, F, CG
- DILU (对角不完全LU)
- Chebyshev

**配置示例** (在 solver_config.yaml 中修改):
```yaml
amgx:
  solver: PCG           # 固定
  preconditioner: AMG   # AMG 或 JACOBI
```

---

## 2. HYPRE (CPU/GPU)

### 当前实现的求解器
- ✅ **PCG** (ParCSR PCG)

### 当前实现的预条件器

| 预条件器 | YAML配置 | 状态 | 说明 |
|---------|----------|------|------|
| **BoomerAMG** | `BOOMERAMG` | ✅ 已实现 | 经典AMG，强烈推荐 |
| **Jacobi** | `JACOBI` | ⏳ 可添加 | 简单预条件器 |
| **无预条件** | `NONE` | ✅ 已实现 | 直接PCG求解 |

### HYPRE 库支持但未实现的选项

**求解器**:
- GMRES
- FlexGMRES  
- BiCGSTAB
- Hybrid (组合求解器)
- LGMRES
- COGMRES

**预条件器**:
- ParaSails (稀疏近似逆)
- Euclid (并行ILU)
- AMS (辅助空间AMG)
- ADS (辅助微分空间)
- MGR (多重网格缩减)
- ILU (不完全LU)
- Schwarz
- BoomerAMG 的其他配置:
  - 不同粗化策略: HMIS, PMIS, CGC等
  - 不同松弛方法: Jacobi, Hybrid GS等
  - W-cycle, F-cycle

**配置示例**:
```yaml
hypre_cpu:  # 或 hypre_gpu
  solver: PCG                  # 固定
  preconditioner: BOOMERAMG    # BOOMERAMG, JACOBI, 或 NONE
```

---

## 3. PETSc (CPU/GPU)

### 当前实现的求解器
- ✅ **CG** (Conjugate Gradient)

### 当前实现的预条件器

| 预条件器 | YAML配置 | 状态 | 说明 |
|---------|----------|------|------|
| **GAMG** | `GAMG` | ✅ 已实现 | 几何代数多重网格 |
| **Jacobi** | `JACOBI` | ✅ 已实现 | 块Jacobi |
| **无预条件** | `NONE` | ✅ 已实现 | 直接CG求解 |

### PETSc 库支持但未实现的选项

**求解器** (KSP):
- GMRES
- DGMRES (偏转GMRES)
- FGMRES (灵活GMRES)
- LGMRES
- BiCGSTAB
- BCGS (BiCG稳定)
- Richardson
- Chebyshev
- GCR
- MINRES
- TFQMR
- 还有30+种其他求解器...

**预条件器** (PC):
- **AMG类**:
  - HYPRE (如果链接HYPRE)
  - ML (Trilinos)
- **直接法**:
  - LU (直接LU分解)
  - Cholesky
  - ILU (不完全LU)
  - ICC (不完全Cholesky)
- **迭代法**:
  - SOR
  - SSOR
  - Eisenstat (优化的SSOR)
  - ASM (加性Schwarz)
  - GASM (广义ASM)
- **其他**:
  - MG (几何多重网格)
  - BDDC (平衡区域分解)
  - FieldSplit (场分裂)
  - Shell (自定义)
  - 还有40+种其他预条件器...

**配置示例**:
```yaml
petsc_cpu:  # 或 petsc_gpu
  solver: CG              # 固定
  preconditioner: GAMG    # GAMG, JACOBI, 或 NONE
```

---

## 🎯 快速参考表

### 求解器对比

| 求解器类型 | AMGX | HYPRE | PETSc | 适用问题 |
|-----------|------|-------|-------|---------|
| **PCG/CG** | ✅ | ✅ | ✅ | 对称正定 |
| **GMRES** | 库支持 | 库支持 | 库支持 | 非对称 |
| **BiCGSTAB** | 库支持 | 库支持 | 库支持 | 非对称 |

### 预条件器效果对比 (Poisson 问题)

| 预条件器 | 实现库 | 迭代数 | 性能 | 推荐度 |
|---------|--------|--------|------|-------|
| **BoomerAMG** | HYPRE | 5 | ⭐⭐⭐⭐⭐ | 强烈推荐 |
| **GAMG** | PETSc | 12 | ⭐⭐⭐⭐ | 推荐 |
| **AMG** | AMGX | 65 | ⭐⭐⭐ | 推荐 |
| **Jacobi** | 所有库 | >60 | ⭐⭐ | 仅测试用 |
| **NONE** | 所有库 | >100 | ⭐ | 不推荐 |

---

## 🔧 如何添加新的求解器/预条件器

### 步骤1: 更新 YAML 配置

编辑 `solver_config.yaml`:
```yaml
amgx:
  solver: GMRES  # 改为新求解器
  preconditioner: DILU  # 改为新预条件器
```

### 步骤2: 更新配置读取器

编辑 `common/config_reader.h`，添加新的getter函数（如需要）

### 步骤3: 更新求解器代码

在对应的 `*_tests/poisson_solver.cpp` 中:

**AMGX 示例**:
```cpp
if (precond_type == "AMG") {
    // AMG 配置
} else if (precond_type == "DILU") {
    // 添加 DILU 配置
    config_str = "{ ... DILU config ... }";
}
```

**HYPRE 示例**:
```cpp
if (precond_type == "BOOMERAMG") {
    // BoomerAMG 配置
} else if (precond_type == "EUCLID") {
    // 添加 Euclid 配置
    HYPRE_EuclidCreate(&precond);
    ...
}
```

**PETSc 示例**:
```cpp
if (precond_type == "GAMG") {
    ierr = PCSetType(pc, PCGAMG);
} else if (precond_type == "ILU") {
    ierr = PCSetType(pc, PCILU);  // 添加 ILU
}
```

### 步骤4: 重新编译

```bash
cd build
make -j
```

---

## 📚 参考文档

### AMGX
- **用户手册**: https://github.com/NVIDIA/AMGX/blob/main/doc/AMGX_Reference.pdf
- **配置示例**: `/home/zzy/Plasma/gpu/amgx/install/lib/configs/`

### HYPRE
- **参考手册**: https://hypre.readthedocs.io/
- **求解器**: https://hypre.readthedocs.io/en/latest/ch-solvers.html
- **预条件器**: https://hypre.readthedocs.io/en/latest/ch-preconditioners.html

### PETSc
- **手册**: https://petsc.org/release/manualpages/
- **KSP**: https://petsc.org/release/manualpages/KSP/
- **PC**: https://petsc.org/release/manualpages/PC/

---

## 💡 推荐配置

### 对称正定问题 (Poisson 方程)

```yaml
# 最优配置
hypre_gpu:
  solver: PCG
  preconditioner: BOOMERAMG

# 或
amgx:
  solver: PCG
  preconditioner: AMG
```

### 非对称问题

可以添加 GMRES + ILU 配置（需要实现）

### 快速测试

```yaml
amgx:
  preconditioner: JACOBI

global:
  problem:
    grid_sizes: [64, 128]  # 小规模
```

---

## 🚀 未来扩展建议

### 高优先级
1. [ ] **GMRES 求解器** - 适用于非对称问题
2. [ ] **ILU 预条件器** - 经典选择
3. [ ] **BiCGSTAB 求解器** - 另一个非对称求解器选项

### 中优先级
4. [ ] **HYPRE ParaSails** - 稀疏近似逆
5. [ ] **PETSc Hypre接口** - 在PETSc中使用HYPRE
6. [ ] **不同AMG配置** - 测试各种粗化和平滑策略

### 低优先级
7. [ ] **直接求解器** - LU, Cholesky
8. [ ] **更多平滑器** - GS, SOR等

---

## 📝 当前限制

1. **求解器固定**
   - 所有模块当前只支持 CG/PCG
   - 硬编码在源代码中
   - 可以扩展为 YAML 可配置

2. **预条件器参数固定**
   - AMG 参数（粗化、平滑等）在代码中
   - 可以扩展到 YAML 配置

3. **GPU 模式**
   - HYPRE/PETSc 的 GPU 模式在代码中设置
   - 可以扩展到 YAML 配置

---

**总结**: 当前实现专注于 Poisson 问题的 CG/PCG 求解器，配备了最常用的预条件器选项。可以根据需要轻松扩展。

