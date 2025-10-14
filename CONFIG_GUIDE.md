# YAML 配置系统使用指南

**创建日期**: 2025-10-14  
**状态**: ✅ 所有5个模块已支持

---

## 📄 配置文件

配置文件位置: **`solver_config.yaml`**

---

## 🎯 支持的配置选项

### 1. AMGX

**可选预条件器**:
- `AMG` - 代数多重网格 (推荐，快4倍)
- `JACOBI` - 块Jacobi (简单但慢)

**配置示例**:
```yaml
amgx:
  solver: PCG
  preconditioner: AMG  # 或 JACOBI
```

**效果对比** (512×512):
| 预条件器 | 迭代数 | 求解时间 | 总时间 |
|---------|--------|---------|--------|
| AMG | 65 | 0.112s | 0.147s ⚡ |
| JACOBI | 61 | 0.636s | 0.639s |

---

### 2. HYPRE-CPU

**可选预条件器**:
- `BOOMERAMG` - BoomerAMG 预条件器 (推荐)
- `JACOBI` - Jacobi 预条件器
- `NONE` - 无预条件器

**配置示例**:
```yaml
hypre_cpu:
  solver: PCG
  preconditioner: BOOMERAMG  # 或 JACOBI, NONE
```

---

### 3. HYPRE-GPU

**可选预条件器**:
- `BOOMERAMG` - BoomerAMG 预条件器 (推荐)
- `JACOBI` - Jacobi 预条件器
- `NONE` - 无预条件器

**配置示例**:
```yaml
hypre_gpu:
  solver: PCG
  preconditioner: BOOMERAMG  # 或 JACOBI, NONE
```

---

### 4. PETSc-CPU

**可选预条件器**:
- `GAMG` - 几何代数多重网格 (推荐)
- `JACOBI` - Jacobi 预条件器
- `NONE` - 无预条件器

**配置示例**:
```yaml
petsc_cpu:
  solver: CG
  preconditioner: GAMG  # 或 JACOBI, NONE
```

---

### 5. PETSc-GPU

**可选预条件器**:
- `GAMG` - 几何代数多重网格 (推荐)
- `JACOBI` - Jacobi 预条件器
- `NONE` - 无预条件器

**配置示例**:
```yaml
petsc_gpu:
  solver: CG
  preconditioner: GAMG  # 或 JACOBI, NONE
```

---

## 🚀 快速使用

### 方法1: 修改配置文件（推荐）

1. **编辑配置文件**
   ```bash
   nano solver_config.yaml
   ```

2. **修改预条件器**
   ```yaml
   amgx:
     preconditioner: JACOBI  # 改为 AMG
   ```

3. **运行测试**（无需重新编译）
   ```bash
   cd build
   ./amgx_tests/amgx_poisson_solver
   ```

### 方法2: 测试不同配置

```bash
# 1. 测试 AMGX AMG
sed -i 's/preconditioner: JACOBI/preconditioner: AMG/' solver_config.yaml
./build/amgx_tests/amgx_poisson_solver

# 2. 测试 AMGX JACOBI
sed -i 's/preconditioner: AMG/preconditioner: JACOBI/' solver_config.yaml
./build/amgx_tests/amgx_poisson_solver
```

---

## 📊 全局配置

### 网格规模

修改所有测试的网格规模:
```yaml
global:
  problem:
    grid_sizes: [64, 128, 256, 512, 1024]  # 添加 1024
```

### 输出设置

```yaml
global:
  output:
    results_dir: results
    print_residual: true
    save_csv: true
    save_json: true
```

---

## 🔍 配置示例场景

### 场景1: 快速测试（小规模 + 简单预条件器）

```yaml
amgx:
  preconditioner: JACOBI

hypre_cpu:
  preconditioner: NONE

petsc_cpu:
  preconditioner: JACOBI

global:
  problem:
    grid_sizes: [64, 128]  # 只测试小规模
```

### 场景2: 性能测试（大规模 + 最优预条件器）

```yaml
amgx:
  preconditioner: AMG

hypre_cpu:
  preconditioner: BOOMERAMG

hypre_gpu:
  preconditioner: BOOMERAMG

petsc_cpu:
  preconditioner: GAMG

petsc_gpu:
  preconditioner: GAMG

global:
  problem:
    grid_sizes: [256, 512, 1024, 2048]  # 大规模
```

### 场景3: 预条件器对比

测试同一库的不同预条件器:

**测试1**: PETSc GAMG
```yaml
petsc_cpu:
  preconditioner: GAMG
```

**测试2**: PETSc Jacobi
```yaml
petsc_cpu:
  preconditioner: JACOBI
```

**测试3**: PETSc 无预条件
```yaml
petsc_cpu:
  preconditioner: NONE
```

---

## ✅ 验证配置

运行任一测试程序，会在开始时打印配置信息:

```
Configuration for AMGX:
  Config file: ../solver_config.yaml
  Solver: PCG
  Preconditioner: AMG
  Grid sizes: 64, 128, 256, 512
```

确认配置正确后再继续测试。

---

## 💡 高级技巧

### 快速切换预条件器

创建多个配置文件:
```bash
# 创建不同配置
cp solver_config.yaml solver_config_amg.yaml
cp solver_config.yaml solver_config_jacobi.yaml

# 编辑各自的预条件器设置
...

# 使用时指定配置文件（需修改代码支持）
./amgx_poisson_solver solver_config_amg.yaml
```

### 批量测试脚本

```bash
#!/bin/bash
# test_all_precond.sh

for precond in AMG JACOBI; do
  echo "Testing with $precond"
  sed -i "s/preconditioner:.*/preconditioner: $precond/" solver_config.yaml
  ./build/amgx_tests/amgx_poisson_solver
  mv results/amgx_results.csv results/amgx_${precond}_results.csv
done
```

---

## 📝 注意事项

1. **配置文件位置**: 必须在 `build/` 的上一级目录（即项目根目录）
2. **无需重新编译**: 修改 YAML 后直接运行即可
3. **大小写敏感**: 预条件器名称必须大写（`AMG` 不是 `amg`）
4. **默认值**: 如果配置文件读取失败，会使用代码中的默认值

---

## 🎯 当前默认配置

```yaml
amgx: AMG
hypre_cpu: BOOMERAMG
hypre_gpu: BOOMERAMG
petsc_cpu: GAMG
petsc_gpu: GAMG
```

---

**配置文件**: `solver_config.yaml`  
**读取器**: `common/config_reader.h`  
**最后更新**: 2025-10-14

