# LinearAlgebra_cuda 项目状态

**最后更新**：2025-10-13

## ✅ 编译状态

**编译成功！** 项目已成功构建，可以使用。

```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda
./build.sh
```

## 📊 测试程序状态

| 库名称 | 状态 | 可执行文件 | 说明 |
|--------|------|-----------|------|
| **AMGX** | ✅ 可用 | `build/amgx_tests/amgx_poisson_solver` | 完全实现并测试 |
| **HYPRE** | ✅ 可用 | `build/hypre_tests/hypre_poisson_solver` | 完全实现并测试 |
| **PETSc** | ✅ 可用 | `build/petsc_tests/petsc_poisson_solver` | 完全实现并测试 |

## 🚀 快速开始

### 运行 AMGX 测试

```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda/build
./amgx_tests/amgx_poisson_solver
```

**测试内容：**
- 求解器：PCG + AMG 预条件器
- 网格规模：64×64, 128×128, 256×256, 512×512
- 输出：`../results/amgx_results.csv` 和 `amgx_results.json`

### 运行 HYPRE 测试

```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda/build
# 单进程
./hypre_tests/hypre_poisson_solver

# 或使用 MPI 多进程
mpirun -np 2 ./hypre_tests/hypre_poisson_solver
```

**测试内容：**
- 求解器：PCG + BoomerAMG 预条件器
- 网格规模：64×64, 128×128, 256×256, 512×512
- 输出：`../results/hypre_results.csv` 和 `hypre_results.json`

## 📁 项目结构

```
LinearAlgebra_cuda/
├── README.md              # 详细文档
├── QUICKSTART.md          # 快速开始指南
├── STATUS.md              # 本文件 - 项目状态
├── build.sh               # 自动构建脚本
├── run_benchmarks.sh      # 自动运行脚本
│
├── common/                # ✅ 通用工具库（已完成）
│   ├── timer.h            # 高精度计时器
│   ├── matrix_generator.h # 测试矩阵生成器
│   └── result_writer.h    # 结果输出工具
│
├── amgx_tests/            # ✅ AMGX 测试（已完成）
│   ├── CMakeLists.txt
│   └── poisson_solver.cpp # 2D Poisson 求解器
│
├── hypre_tests/           # ✅ HYPRE 测试（已完成）
│   ├── CMakeLists.txt
│   └── poisson_solver.cpp # 2D Poisson 求解器
│
├── petsc_tests/           # ✅ PETSc 测试（已完成）
│   ├── CMakeLists.txt
│   └── poisson_solver.cpp # 2D Poisson 求解器
│
├── build/                 # 构建目录
└── results/               # 结果输出目录
```

## 🔧 技术细节

### AMGX 配置
- 求解器：PCG（预条件共轭梯度法）
- 预条件器：AMG（代数多重网格）
- 聚合策略：AGGREGATION
- 平滑器：BLOCK_JACOBI
- 收敛容差：1e-6
- 最大迭代次数：1000

### HYPRE 配置
- 求解器：ParCSR PCG
- 预条件器：BoomerAMG
- 粗化策略：Falgout coarsening (type 6)
- 松弛方法：Symmetric Gauss-Seidel (type 6)
- 收敛容差：1e-6
- 最大迭代次数：1000

## ✅ 问题已解决

### PETSc MPI 版本冲突 - 已解决

**解决方案**：重新编译 PETSc 并安装到独立目录

新编译的 PETSc 位于：`/home/zzy/Plasma/gpu/petsc-gpu/install`

配置详情：
- CUDA 支持：已启用 (CUDA 12.6)
- MPI 支持：OpenMPI 4.0.3
- GPU 架构：70, 75, 80, 86, 89, 90
- 优化级别：-O3 -march=native
- 安装方式：独立安装，避免版本冲突

## 📈 性能测试结果

测试结果会自动保存在 `results/` 目录：

- **CSV 格式**：便于在 Excel 或 Pandas 中分析
- **JSON 格式**：便于程序化处理

### 结果字段

| 字段 | 说明 |
|------|------|
| Library | 库名称（AMGX/HYPRE/PETSc） |
| Test | 测试名称 |
| ProblemSize | 问题规模（矩阵维度） |
| SetupTime | Setup 时间（秒） |
| SolveTime | 求解时间（秒） |
| TotalTime | 总时间（秒） |
| Iterations | 迭代次数 |
| Residual | 最终残差 |

## 🎯 下一步计划

- [x] ✅ 完成 AMGX 实现和测试
- [x] ✅ 完成 HYPRE 实现和测试
- [x] ✅ 解决 PETSc MPI 版本冲突（重新编译）
- [x] ✅ 完成 PETSc 实现和测试
- [ ] 📊 添加 3D Poisson 问题测试
- [ ] ⚡ 添加多 GPU 可扩展性测试
- [ ] 📉 添加不同预条件器比较
- [ ] 📊 生成性能比较图表
- [ ] 📝 添加详细性能分析报告

## 🛠️ 自定义测试

### 修改测试规模

编辑 `amgx_tests/poisson_solver.cpp` 或 `hypre_tests/poisson_solver.cpp`：

```cpp
// 找到这一行
std::vector<int> grid_sizes = {64, 128, 256, 512};

// 修改为你想要的规模
std::vector<int> grid_sizes = {32, 64, 128, 256, 512, 1024, 2048};
```

### 修改求解器参数

AMGX：编辑 `amgx_tests/poisson_solver.cpp` 中的 `config_string`

HYPRE：编辑 `hypre_tests/poisson_solver.cpp` 中的求解器设置代码

## 📞 支持

如遇到问题，请检查：
1. CUDA 版本：12.6+
2. GPU 驱动正常
3. 足够的 GPU 内存
4. 库路径正确配置

---

**项目创建**：2025-10-13
**最后成功构建**：2025-10-13

