# 文档导航

**LinearAlgebra_cuda 项目文档中心**

---

## 📚 文档列表

### 🚀 入门文档

| 文档 | 描述 | 优先级 |
|------|------|--------|
| **[QUICKSTART.md](QUICKSTART.md)** | 5分钟快速上手指南 | ⭐⭐⭐ 必读 |
| **[STATUS.md](STATUS.md)** | 项目当前状态和已知问题 | ⭐⭐ |

### 📊 测试结果归档

| 目录 | 描述 | 优先级 |
|------|------|--------|
| **[test_results/](test_results/)** | 按日期归档的测试数据 | ⭐⭐⭐ |
| **[test_results/2025-10-14/](test_results/2025-10-14/)** | 2025-10-14 测试数据 | ⭐⭐⭐ 最新 |
| **[archived_docs/](archived_docs/)** | 按日期归档的文档快照 | ⭐⭐ |
| **[archived_docs/2025-10-14/](archived_docs/2025-10-14/)** | 2025-10-14 文档快照 | ⭐⭐ 最新 |

**最新测试文档** (2025-10-14):
- [性能对比报告](archived_docs/2025-10-14/PERFORMANCE_COMPARISON.md)
- [求解器配置](archived_docs/2025-10-14/SOLVER_CONFIGURATIONS.md)
- [最终总结](archived_docs/2025-10-14/FINAL_SUMMARY.md)

### 📖 通用项目文档

| 文档 | 描述 | 优先级 |
|------|------|--------|
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | 项目概述和成就统计 | ⭐ |

---

## 🎯 阅读建议

### 如果您是第一次使用
1. 先读 **[QUICKSTART.md](QUICKSTART.md)** - 了解如何运行
2. 再读 **[PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md)** - 了解性能表现

### 如果您想查看性能测试结果
1. 查看测试数据: **[test_results/2025-10-14/](test_results/2025-10-14/)** (CSV/JSON)
2. 阅读性能分析: **[archived_docs/2025-10-14/PERFORMANCE_COMPARISON.md](archived_docs/2025-10-14/PERFORMANCE_COMPARISON.md)**

### 如果您想了解配置细节
1. 阅读 **[archived_docs/2025-10-14/SOLVER_CONFIGURATIONS.md](archived_docs/2025-10-14/SOLVER_CONFIGURATIONS.md)**
2. 查看源代码中的配置部分或 `solver_config.yaml`

### 如果您想了解项目全貌
1. 阅读 **[archived_docs/2025-10-14/FINAL_SUMMARY.md](archived_docs/2025-10-14/FINAL_SUMMARY.md)**
2. 浏览 **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**

---

## 📊 性能数据文件

性能测试结果保存在 `../results/` 目录：

```
results/
├── amgx_results.csv/json           # AMGX GPU
├── hypre_results.csv/json          # HYPRE CPU
├── hypre_gpu_results.csv/json      # HYPRE GPU
├── petsc_results.csv/json          # PETSc CPU
└── petsc_gpu_results.csv/json      # PETSc GPU
```

---

## 🔗 快速链接

### 运行测试
```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda
./build.sh              # 编译
./run_benchmarks.sh     # 运行所有测试
```

### 查看结果
```bash
cd results
cat *.csv              # 查看所有 CSV 结果
```

### 阅读性能报告
```bash
cd doc
cat PERFORMANCE_COMPARISON.md
```

---

**文档更新日期**: 2025-10-13  
**文档数量**: 6个核心文档
