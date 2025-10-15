# LinearAlgebra_cuda - GPU 加速线性代数库性能测试平台

**版本**: 2.0  
**更新日期**: 2025-10-14  
**状态**: ✅ 完成并可用

---

## 🎯 项目简介

这是一个完整的 GPU 加速线性代数库性能测试平台，用于对比 **AMGX**、**HYPRE** 和 **PETSc** 三大主流库在 CPU 和 GPU 模式下的性能表现。

**特点**:
- ✅ **7个成功测试模块** (AMGX, HYPRE, PETSc 的 CPU/GPU 版本 + GPU Native)
- ✅ **GPU Native 实现** (AMGX ⭐, PETSc ⚠️; HYPRE ❌ 有已知问题)
- ✅ **CPU/GPU 双模式对比**
- ✅ **精度完全对齐** (~1e-7 相对残差)
- ✅ **YAML 配置系统** (动态选择求解器和预条件器)
- ✅ **详细性能分析** (7个 Markdown 文档)

---

## 🏆 性能亮点 (512×512 网格)

| 排名 | 库 + 配置 | 总时间 | 求解时间 | 迭代数 |
|-----|----------|--------|---------|--------|
| 🥇 | **AMGX Native + Jacobi** | **0.016s** ⚡ | 0.012s | 1 |
| 🥈 | AMGX Native + AMG | 0.122s | 0.089s | 64 |
| 🥉 | AMGX GPU + AMG | 0.147s | 0.112s | 65 |
| 4 | PETSc GPU + GAMG | 0.452s | **0.034s** | 12 |
| 5 | HYPRE GPU + BoomerAMG | 0.531s | 0.134s | **5** |

**关键发现**:
- ⚡ **AMGX Native + Jacobi** 对 Poisson 问题惊人地快 (1次迭代！)
- 🏆 **HYPRE BoomerAMG** 迭代次数最少 (5次)
- 🎯 **PETSc GPU** 求解阶段最快 (0.034s)

---

## 🚀 快速开始

### 1. 构建项目

**方法 A: 使用 Python 脚本（推荐）⭐**
```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda
python3 rebuild.py
```

**方法 B: 使用传统脚本**
```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda
./build.sh
```

### 2. 运行测试
```bash
cd build

# 标准测试（推荐）
./amgx_tests/amgx_poisson_solver              # AMGX GPU
./hypre_tests/hypre_poisson_solver            # HYPRE CPU
./hypre_gpu_tests/hypre_gpu_poisson_solver    # HYPRE GPU
./petsc_tests/petsc_poisson_solver            # PETSc CPU
./petsc_gpu_tests/petsc_gpu_poisson_solver    # PETSc GPU

# GPU Native 测试（矩阵在GPU端组装）
./amgx_gpu_native_tests/amgx_gpu_native_poisson_solver   # ✅ 推荐
./petsc_gpu_native_tests/petsc_gpu_native_poisson_solver # ⚠️ 比标准版慢
```

### 3. 查看结果
```bash
cd results
ls -lh  # 查看生成的 CSV 和 JSON 文件
```

---

## 📚 文档导航

### ⭐ 核心文档（必读）

| 文档 | 内容简介 | 优先级 |
|------|---------|--------|
| **[2025-10-14_性能测试总结.md](2025-10-14_性能测试总结.md)** | 所有模块完整对比、性能排名、使用建议 | ⭐⭐⭐⭐⭐ |
| **[2025-10-14_AMGX预条件器对比.md](2025-10-14_AMGX预条件器对比.md)** | AMG vs Jacobi 对比，为何 Jacobi 1次迭代 | ⭐⭐⭐⭐ |
| **[solver_config.yaml](solver_config.yaml)** | YAML 配置文件（动态选择求解器） | ⭐⭐⭐ |

### 📖 技术文档

| 文档 | 内容简介 | 优先级 |
|------|---------|--------|
| **[2025-10-14_GPU_Native性能分析.md](2025-10-14_GPU_Native性能分析.md)** | GPU Native 实现对比和分析 | ⭐⭐⭐⭐ |
| **[2025-10-14_GPU实现细节分析.md](2025-10-14_GPU实现细节分析.md)** | 矩阵/向量在 device 还是 host | ⭐⭐⭐ |
| **[2025-10-14_GPU矩阵组装指南.md](2025-10-14_GPU矩阵组装指南.md)** | 如何在 GPU 端组装矩阵 | ⭐⭐⭐ |
| **[2025-10-14_HYPRE问题总结.md](2025-10-14_HYPRE问题总结.md)** | HYPRE GPU Native 失败分析 | ⭐⭐ |

### 📁 历史归档

| 目录 | 内容 |
|------|------|
| **[doc/test_results/2025-10-14/](doc/test_results/2025-10-14/)** | 2025-10-14 测试数据（CSV/JSON） |
| **[doc/archived_docs/2025-10-14/](doc/archived_docs/2025-10-14/)** | 2025-10-14 文档快照 |

---

## 🎯 阅读指南

### 首次使用者
1. **阅读**: [2025-10-14_性能测试总结.md](2025-10-14_性能测试总结.md)
2. **运行**: 上方"快速开始"中的测试
3. **查看**: `results/` 目录下的结果文件

### 想配置求解器
1. **阅读**: [2025-10-14_AMGX预条件器对比.md](2025-10-14_AMGX预条件器对比.md)
2. **编辑**: [solver_config.yaml](solver_config.yaml)
3. **重跑**: 测试并对比性能

### 想了解 GPU Native
1. **阅读**: [2025-10-14_GPU_Native性能分析.md](2025-10-14_GPU_Native性能分析.md)
2. **参考**: [2025-10-14_GPU矩阵组装指南.md](2025-10-14_GPU矩阵组装指南.md)
3. **了解**: [2025-10-14_HYPRE问题总结.md](2025-10-14_HYPRE问题总结.md)

### 深入技术细节
1. **阅读**: [2025-10-14_GPU实现细节分析.md](2025-10-14_GPU实现细节分析.md)
2. **浏览**: 源代码 (`*_tests/` 目录)

---

## 📦 测试模块

```
LinearAlgebra_cuda/
├── amgx_tests/              # AMGX GPU (标准)
├── amgx_gpu_native/         # AMGX GPU Native ✅
├── hypre_tests/             # HYPRE CPU
├── hypre_gpu_tests/         # HYPRE GPU
├── hypre_gpu_native/        # HYPRE GPU Native ❌
├── petsc_tests/             # PETSc CPU
├── petsc_gpu_tests/         # PETSc GPU
├── petsc_gpu_native/        # PETSc GPU Native ⚠️
├── common/                  # 通用工具库
│   ├── timer.h              # 计时器
│   ├── result_writer.h      # 结果输出
│   ├── matrix_generator.h   # 矩阵生成
│   ├── config_reader.h      # YAML 配置读取
│   └── gpu_matrix_generator.cuh  # GPU 矩阵生成核函数
├── build/                   # 构建目录
└── results/                 # 结果输出目录
```

---

## 🛠️ 环境要求

- **CUDA**: 12.6+
- **GPU**: NVIDIA (测试环境: RTX 4060 Ti)
- **MPI**: OpenMPI 3.1+
- **CMake**: 3.18+
- **编译器**: GCC 9.4+
- **Python**: 3.6+ (用于 rebuild.py)
- **yaml-cpp**: 用于配置文件解析

---

## 🔄 换电脑/迁移

**只需 3 步**：

```bash
# 1. 修改库路径
vim env_setup.sh  # 编辑第 10-20 行

# 2. 加载环境
source env_setup.sh

# 3. 重新构建
python3 rebuild.py --use-env
```

**详细说明**: 查看 [使用说明_换电脑.md](使用说明_换电脑.md)

---

## 💡 使用建议

### 根据场景选择库

**Poisson 类问题（对角占优）**:
```yaml
amgx:
  preconditioner: JACOBI  # 惊人地快！1次迭代
```
→ 使用 `amgx_gpu_native_tests/amgx_gpu_native_poisson_solver`

**复杂问题（各向异性、病态）**:
```yaml
hypre_gpu:
  preconditioner: BOOMERAMG  # 迭代最少
```
→ 使用 `hypre_gpu_tests/hypre_gpu_poisson_solver`

**生产环境（需要可靠性）**:
```yaml
petsc_gpu:
  preconditioner: GAMG  # 成熟稳定
```
→ 使用 `petsc_gpu_tests/petsc_gpu_poisson_solver`

---

## 📊 性能数据文件

所有测试结果保存在 `results/` 目录：

```
results/
├── amgx_results.csv / .json
├── amgx_gpu_native_results.csv / .json  ⭐ 最快
├── hypre_results.csv / .json
├── hypre_gpu_results.csv / .json
├── petsc_results.csv / .json
├── petsc_gpu_results.csv / .json
└── petsc_gpu_native_results.csv / .json
```

---

## 🔧 配置文件

编辑 **[solver_config.yaml](solver_config.yaml)** 可以动态切换：

- 网格尺寸
- 求解器类型
- 预条件器类型
- 预条件器参数

无需重新编译！

---

## 📝 更新日志

### 2025-10-14 (v2.0)

**新增**:
- ✅ GPU Native 实现 (AMGX, PETSc)
- ✅ YAML 配置系统
- ✅ 动态预条件器选择
- ✅ 详细技术文档 (7个)

**发现**:
- ⚡ AMGX + Jacobi 对 Poisson 问题 1 次迭代收敛
- 📊 GPU Native 对 AMGX 提升 9.2 倍
- ❌ HYPRE GPU Native 在 WSL2 环境下有兼容性问题

**优化**:
- 精度对齐 (所有库 ~1e-7)
- 文档重组和重命名
- 删除冗余文档

### 2025-10-13 (v1.0)

- 初始版本
- 5个测试模块
- 基础性能对比

---

**项目路径**: `/home/zzy/Plasma/gpu/LinearAlgebra_cuda`  
**维护状态**: ✅ 活跃  
**测试问题**: 2D Poisson 方程 (5点模板)
