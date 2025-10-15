# doc/ 目录说明

**更新日期**: 2025-10-14

---

## 📂 目录结构

```
doc/
├── README.md                    # 本文档
└── test_results/                # 测试数据归档
    ├── README.md
    └── 2025-10-14/              # 2025-10-14 的测试数据
        ├── amgx_results.csv
        ├── amgx_results.json
        ├── hypre_results.csv
        ├── hypre_results.json
        ├── hypre_gpu_results.csv
        ├── hypre_gpu_results.json
        ├── petsc_results.csv
        ├── petsc_results.json
        ├── petsc_gpu_results.csv
        ├── petsc_gpu_results.json
        └── README.md
```

---

## 📚 主要文档位置

**所有核心文档已移至项目根目录**，便于快速访问：

```
LinearAlgebra_cuda/
├── README.md                            ← 项目主页 ⭐⭐⭐⭐⭐
├── 2025-10-14_性能测试总结.md           ← 性能对比 ⭐⭐⭐⭐⭐
├── 2025-10-14_AMGX预条件器对比.md       ← AMGX 分析 ⭐⭐⭐⭐
├── 2025-10-14_GPU_Native性能分析.md    ← GPU Native ⭐⭐⭐⭐
├── 2025-10-14_GPU实现细节分析.md       ← 实现细节 ⭐⭐⭐
├── 2025-10-14_GPU矩阵组装指南.md       ← 组装指南 ⭐⭐⭐
├── 2025-10-14_HYPRE问题总结.md          ← HYPRE 问题 ⭐⭐
├── solver_config.yaml                   ← 配置文件 ⭐⭐⭐
└── 文档说明.md                          ← 文档索引 ⭐⭐
```

**请返回根目录查看最新文档！**

---

## 📊 test_results/ 说明

此目录仅用于**归档测试数据**（CSV 和 JSON 文件）。

### 当前归档

- **2025-10-14**: 包含 5 个库（AMGX, HYPRE-CPU, HYPRE-GPU, PETSc-CPU, PETSc-GPU）的测试数据

### 未来归档

如果进行新的测试，可以创建新的日期目录：
```
test_results/
├── 2025-10-14/    # 当前测试
├── 2025-10-15/    # 未来测试
└── ...
```

---

## 🗑️ 已清理的内容

以下内容已被清理（内容过时或重复）：

### 删除的目录
- ❌ `doc/archived_docs/` - 旧文档快照，已被根目录最新文档取代

### 删除的文件
- ❌ `doc/PROJECT_SUMMARY.md` - 过时，新版在根目录
- ❌ `doc/QUICKSTART.md` - 基于旧的 5 模块，已过时
- ❌ `doc/STATUS.md` - 过时

---

## 🎯 如何使用

### 查看最新文档
返回项目根目录：
```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda
ls -lh *.md
```

### 查看测试数据
```bash
cd doc/test_results/2025-10-14/
ls -lh
```

### 快速导航
- **项目主页**: `../README.md`
- **性能总结**: `../2025-10-14_性能测试总结.md`
- **配置文件**: `../solver_config.yaml`

---

**维护原则**: 
- 📄 文档放在根目录，便于访问
- 📊 数据放在 `doc/test_results/`，按日期归档
- 🗑️ 定期清理过时文档，避免重复
