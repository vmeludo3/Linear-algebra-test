# 测试结果归档

这个目录存储历史测试结果，按日期组织。

---

## 归档结构

```
test_results/
├── README.md           # 本文件
├── 2025-10-14/         # 2025年10月14日的测试
│   ├── README.md       # 测试说明
│   ├── *.csv           # CSV 结果
│   └── *.json          # JSON 结果
└── (未来日期)/         # 未来的测试...
```

---

## 已归档的测试

| 日期 | 测试模块数 | 数据点数 | 说明 |
|------|-----------|---------|------|
| **2025-10-14** | 5 | 20 | 初始完整测试，所有模块成功 |

---

## 如何使用

### 查看特定日期的结果
```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda/doc/test_results/2025-10-14
cat *.csv
```

### 对比不同日期的结果
```bash
# 使用 diff 或自定义脚本对比
diff 2025-10-14/amgx_results.csv 2025-10-15/amgx_results.csv
```

### 归档新的测试结果
```bash
# 创建新日期目录
mkdir -p test_results/YYYY-MM-DD

# 复制结果
cp ../../results/* test_results/YYYY-MM-DD/

# 添加说明文档
nano test_results/YYYY-MM-DD/README.md
```

---

## 数据文件格式

### CSV 格式
```csv
Library,Test,ProblemSize,SetupTime(s),SolveTime(s),TotalTime(s),Iterations,Residual
```

### JSON 格式
```json
{
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "results": [...]
}
```

---

**最后更新**: 2025-10-14

