# LinearAlgebra_cuda 快速开始指南

## 📁 项目结构

```
LinearAlgebra_cuda/
├── README.md              # 详细说明文档
├── QUICKSTART.md          # 本文件 - 快速开始
├── CMakeLists.txt         # 主 CMake 配置
├── build.sh               # 自动构建脚本
├── run_benchmarks.sh      # 自动运行所有测试
├── .gitignore             # Git 忽略文件
│
├── common/                # 通用工具库
│   ├── CMakeLists.txt
│   ├── timer.h            # 高精度计时器
│   ├── matrix_generator.h # 测试矩阵生成器
│   └── result_writer.h    # 结果输出工具
│
├── amgx_tests/            # AMGX 测试
│   ├── CMakeLists.txt
│   └── poisson_solver.cpp # AMGX Poisson 求解器
│
├── hypre_tests/           # HYPRE 测试
│   ├── CMakeLists.txt
│   └── poisson_solver.cpp # HYPRE Poisson 求解器 (待完善)
│
├── petsc_tests/           # PETSc 测试
│   ├── CMakeLists.txt
│   └── poisson_solver.cpp # PETSc Poisson 求解器 (待完善)
│
├── build/                 # CMake 构建目录
└── results/               # 测试结果输出目录
```

## 🚀 快速开始

### 方法 1: 使用自动脚本（推荐）

```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda

# 构建项目
./build.sh

# 运行测试
./run_benchmarks.sh
```

### 方法 2: 手动构建

```bash
cd /home/zzy/Plasma/gpu/LinearAlgebra_cuda

# 创建并进入 build 目录
mkdir -p build && cd build

# 配置 CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DHYPRE_DIR=/home/zzy/Plasma/gpu/hypre \
    -DPETSC_DIR=/home/zzy/Plasma/gpu/ltpDeps-v2412/extract/petsc-3.22.2 \
    -DAMGX_DIR=/home/zzy/Plasma/gpu/amgx/install

# 编译
make -j$(nproc)

# 运行 AMGX 测试
./amgx_tests/amgx_poisson_solver
```

## 🧪 可用测试

### 1. AMGX Poisson 求解器 ✅
已完成实现，测试不同规模的 2D Poisson 问题。

**运行：**
```bash
cd build
./amgx_tests/amgx_poisson_solver
```

**测试内容：**
- 网格规模：64×64, 128×128, 256×256, 512×512
- 求解器：PCG + AMG 预条件器
- 输出：性能指标、CSV 和 JSON 结果文件

### 2. HYPRE Poisson 求解器 🚧
框架已建立，待完善实现。

### 3. PETSc Poisson 求解器 🚧
框架已建立，待完善实现。

## 📊 结果输出

测试结果会自动保存在 `results/` 目录：

- `amgx_results.csv` - CSV 格式结果（可用 Excel 打开）
- `amgx_results.json` - JSON 格式结果（结构化数据）
- 类似的文件用于 HYPRE 和 PETSc

### 结果文件格式

**CSV 格式：**
```
Library,Test,ProblemSize,SetupTime(s),SolveTime(s),TotalTime(s),Iterations,Residual
AMGX,Poisson_2D_PCG_AMG,4096,0.123456,0.234567,0.358023,42,1.23e-07
```

**JSON 格式：**
```json
{
  "timestamp": "2025-10-13 15:30:00",
  "results": [
    {
      "library": "AMGX",
      "test": "Poisson_2D_PCG_AMG",
      "problem_size": 4096,
      "setup_time": 0.123456,
      "solve_time": 0.234567,
      "total_time": 0.358023,
      "iterations": 42,
      "residual": 1.23e-07
    }
  ]
}
```

## 🔧 自定义测试

### 修改测试规模

编辑 `amgx_tests/poisson_solver.cpp`：

```cpp
// 在 main 函数中找到这一行
std::vector<int> grid_sizes = {64, 128, 256, 512};

// 修改为你想要的规模，例如：
std::vector<int> grid_sizes = {32, 64, 128, 256, 512, 1024};
```

### 修改求解器配置

在 `amgx_tests/poisson_solver.cpp` 中修改 `config_string`：

```cpp
const char* config_string = 
    "{\n"
    "  \"config_version\": 2,\n"
    "  \"solver\": {\n"
    "    \"solver\": \"GMRES\",      // 改用 GMRES
    "    \"max_iters\": 2000,        // 增加最大迭代次数
    "    \"tolerance\": 1e-8,        // 更严格的收敛条件
    ...
```

## 📈 性能分析工具

### 通用工具类

**BenchmarkTimer (timer.h):**
```cpp
BenchmarkTimer timer;
timer.start("my_operation");
// ... 执行操作 ...
timer.stop("my_operation");
timer.print_summary();  // 打印所有计时结果
```

**MatrixGenerator (matrix_generator.h):**
```cpp
std::vector<int> rows, cols;
std::vector<double> values, rhs;

// 生成 2D Poisson 矩阵
MatrixGenerator::generate_2d_poisson_5pt(nx, ny, rows, cols, values, rhs);

// 生成 3D Poisson 矩阵
MatrixGenerator::generate_3d_poisson_7pt(nx, ny, nz, rows, cols, values, rhs);
```

**ResultWriter (result_writer.h):**
```cpp
ResultWriter writer("../results");

ResultWriter::BenchmarkResult result;
result.library_name = "AMGX";
result.test_name = "MyTest";
result.problem_size = 10000;
result.solve_time = 1.234;
// ... 设置其他字段 ...

writer.add_result(result);
writer.write_csv("my_results.csv");
writer.write_json("my_results.json");
writer.print_summary();
```

## 🎯 下一步

1. **运行 AMGX 测试**：验证基本框架工作正常
2. **完善 HYPRE 测试**：实现 BoomerAMG + PCG 求解器
3. **完善 PETSc 测试**：实现 GAMG + KSP 求解器
4. **添加更多测试**：
   - 3D Poisson 问题
   - 不同预条件器比较
   - 多 GPU 可扩展性测试
   - 不同稀疏模式的矩阵

## 💡 提示

- 首次运行可能需要较长时间编译
- 确保 GPU 驱动正常工作
- 大规模问题可能需要较多 GPU 内存
- 使用 `nvidia-smi` 监控 GPU 使用情况

## 📝 已安装的库

✅ **HYPRE**: `/home/zzy/Plasma/gpu/hypre`
✅ **PETSc**: `/home/zzy/Plasma/gpu/ltpDeps-v2412/extract/petsc-3.22.2`
✅ **AMGX**: `/home/zzy/Plasma/gpu/amgx/install`

所有库均支持 CUDA GPU 加速！

---

**创建日期**: 2025-10-13

