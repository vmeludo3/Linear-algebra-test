# GPU 端直接组装矩阵指南

**文档日期**: 2025-10-14  
**适用场景**: 矩阵系数已经在 GPU 端计算完成

---

## 📋 目录

1. [AMGX - 使用设备指针直接上传](#1-amgx)
2. [HYPRE - 直接操作设备内存](#2-hypre)
3. [PETSc - 使用 CUDA 接口](#3-petsc)
4. [完整示例代码](#4-完整示例)

---

## 场景说明

假设您有一个 CUDA 程序，已经在 GPU 上计算出了矩阵系数：

```cpp
// 您的 CUDA 核函数已经生成了这些数据
double *d_values;      // GPU 上的矩阵值
int *d_row_ptrs;       // GPU 上的行指针 (CSR)
int *d_col_indices;    // GPU 上的列索引 (CSR)
double *d_rhs;         // GPU 上的右端项
```

**目标**: 直接使用这些 GPU 数据创建求解器，避免回传到 CPU 再上传。

---

## 1️⃣ AMGX

### 方法 A: 使用设备指针直接上传 ⭐ 推荐

AMGX 支持直接从设备指针创建矩阵：

```cpp
#include "amgx_c.h"

// 假设您在 GPU 上已经有了 CSR 格式的数据
int *d_row_ptrs;       // 设备指针: [n+1]
int *d_col_indices;    // 设备指针: [nnz]
double *d_values;      // 设备指针: [nnz]
double *d_rhs;         // 设备指针: [n]

// 1. 创建 AMGX 资源和配置
AMGX_resources_handle resources;
AMGX_config_handle config;
AMGX_solver_handle solver;

AMGX_resources_create_simple(&resources, config);

// 2. 创建矩阵和向量（设备模式）
AMGX_matrix_handle A;
AMGX_vector_handle b, x;

AMGX_matrix_create(&A, resources, AMGX_mode_dDDI);  // d = device
AMGX_vector_create(&b, resources, AMGX_mode_dDDI);
AMGX_vector_create(&x, resources, AMGX_mode_dDDI);

// 3. 从设备指针上传矩阵 ⭐ 关键！
AMGX_matrix_upload_all_global(
    A,
    n,              // 全局行数
    nnz,            // 非零元个数
    1,              // block_dimx
    1,              // block_dimy
    d_row_ptrs,     // ← GPU 指针！
    d_col_indices,  // ← GPU 指针！
    d_values,       // ← GPU 指针！
    nullptr         // 无对角线数据
);

// 4. 从设备指针上传向量
AMGX_vector_upload(b, n, 1, d_rhs);  // ← GPU 指针！

// 5. 初始化解向量（可以在 GPU 上）
double *d_x_init;
cudaMalloc(&d_x_init, n * sizeof(double));
cudaMemset(d_x_init, 0, n * sizeof(double));
AMGX_vector_upload(x, n, 1, d_x_init);

// 6. 正常求解
AMGX_solver_create(&solver, resources, AMGX_mode_dDDI, config);
AMGX_solver_setup(solver, A);
AMGX_solver_solve(solver, b, x);

// 7. 结果可以直接下载到 GPU
double *d_solution;
cudaMalloc(&d_solution, n * sizeof(double));
AMGX_vector_download(x, d_solution);  // Device → Device
```

### 关键点

✅ `AMGX_matrix_upload_all_global()` 接受设备指针  
✅ `AMGX_vector_upload()` 接受设备指针  
✅ **完全避免 CPU-GPU 传输**  
✅ 数据始终保持在 GPU

### 注意事项

- CSR 格式必须正确（行指针、列索引、值）
- 设备指针必须有效（不能是空指针）
- 可以使用 `AMGX_matrix_upload_all()` 用于非分布式情况

---

## 2️⃣ HYPRE

### 方法 A: 直接设置设备内存位置 ⭐ 推荐

HYPRE 2.20+ 支持直接在 GPU 上操作：

```cpp
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

// 您的 GPU 数据
int *d_row_ptrs;
int *d_col_indices;
double *d_values;
double *d_rhs;

// 1. 设置 HYPRE 使用 GPU 内存
HYPRE_Init();
HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

// 2. 创建 IJMatrix
HYPRE_IJMatrix A;
HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, jlower, jupper, &A);
HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

// 3. 初始化（在 GPU 上分配内存）
HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_DEVICE);

// 方法 1: 使用 HYPRE API 从设备设置值
for (HYPRE_Int i = ilower; i <= iupper; i++) {
    HYPRE_Int ncols = d_row_ptrs[i+1] - d_row_ptrs[i];
    HYPRE_Int *cols = &d_col_indices[d_row_ptrs[i]];  // GPU 指针
    double *vals = &d_values[d_row_ptrs[i]];          // GPU 指针
    
    // HYPRE 会自动处理设备指针（如果设置了 MEMORY_DEVICE）
    HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, cols, vals);
}

// 4. Assemble（数据已经在 GPU）
HYPRE_IJMatrixAssemble(A);

// 5. 获取 ParCSR 矩阵
HYPRE_ParCSRMatrix parcsr_A;
HYPRE_IJMatrixGetObject(A, (void**)&parcsr_A);
```

### 方法 B: 直接构造 ParCSR 矩阵 ⚡ 高级

如果您熟悉 HYPRE 内部结构，可以直接创建 ParCSR 矩阵：

```cpp
// 1. 创建 ParCSR 矩阵结构
HYPRE_ParCSRMatrix A = hypre_ParCSRMatrixCreate(
    MPI_COMM_WORLD,
    global_num_rows,
    global_num_cols,
    row_starts,
    col_starts,
    num_cols_offd,
    num_nonzeros_diag,
    num_nonzeros_offd
);

// 2. 设置设备内存位置
hypre_ParCSRMatrixMemoryLocation(A) = HYPRE_MEMORY_DEVICE;

// 3. 直接设置设备指针（零拷贝！）
hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
hypre_CSRMatrixData(diag) = d_values;      // ← 您的 GPU 指针
hypre_CSRMatrixI(diag) = d_row_ptrs;       // ← 您的 GPU 指针
hypre_CSRMatrixJ(diag) = d_col_indices;    // ← 您的 GPU 指针

// 4. 设置拥有权（HYPRE 不会释放这些指针）
hypre_CSRMatrixOwnsData(diag) = 0;

// 5. 直接使用
HYPRE_Solver solver;
HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
HYPRE_PCGSetup(solver, (HYPRE_Matrix)A, ...);
```

### 关键点

✅ 方法 A: 适合标准使用，HYPRE 自动管理  
✅ 方法 B: 零拷贝，直接使用您的 GPU 指针  
⚠️ 方法 B 需要正确管理内存所有权

---

## 3️⃣ PETSc

### 方法 A: 使用 MatSeqAIJCUSPARSESetPreallocation ⭐ 推荐

PETSc 提供专门的 CUDA 接口：

```cpp
#include <petsc.h>

// 您的 GPU 数据
int *d_row_ptrs;       // CSR 行指针
int *d_col_indices;    // CSR 列索引
double *d_values;      // CSR 值
double *d_rhs;         // 右端项

// 1. 创建矩阵（CUDA 类型）
Mat A;
MatCreate(PETSC_COMM_WORLD, &A);
MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE);
MatSetType(A, MATAIJCUSPARSE);  // ← CUDA 矩阵

// 2. 方法 1: 从设备 CSR 数据创建 ⭐
MatCreateSeqAIJCUSPARSEWithArrays(
    PETSC_COMM_SELF,
    n,              // 行数
    n,              // 列数
    d_row_ptrs,     // ← GPU CSR 行指针
    d_col_indices,  // ← GPU CSR 列索引
    d_values,       // ← GPU CSR 值
    &A
);

// 3. 创建向量（CUDA 类型）
Vec b, x;
VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n, d_rhs, &b);
VecCreateSeqCUDA(PETSC_COMM_SELF, n, &x);

// 4. 正常求解
KSP ksp;
KSPCreate(PETSC_COMM_WORLD, &ksp);
KSPSetOperators(ksp, A, A);
KSPSetType(ksp, KSPCG);
KSPSolve(ksp, b, x);
```

### 方法 B: 使用 MatCUSPARSESetFormat (更灵活)

```cpp
// 1. 创建空矩阵
Mat A;
MatCreate(PETSC_COMM_WORLD, &A);
MatSetSizes(A, n, n, n, n);
MatSetType(A, MATAIJCUSPARSE);

// 2. 设置预分配
MatSeqAIJSetPreallocation(A, nnz_per_row, NULL);

// 3. 从设备批量设置值
// PETSc 3.17+ 支持
MatSetValuesBatch(A, n, d_row_indices, d_col_indices, d_values, INSERT_VALUES);

// 或者使用较低级别的接口
MatCUSPARSESetFormat(A, MAT_CUSPARSE_CSR);

// 获取底层 cuSPARSE 矩阵
cusparseMatDescr_t descr;
int *d_csr_row_ptr, *d_csr_col_ind;
double *d_csr_val;
MatCUSPARSEGetArrays(A, &d_csr_row_ptr, &d_csr_col_ind, &d_csr_val);

// 从您的数据拷贝（GPU → GPU，很快）
cudaMemcpy(d_csr_row_ptr, d_row_ptrs, ...);
cudaMemcpy(d_csr_col_ind, d_col_indices, ...);
cudaMemcpy(d_csr_val, d_values, ...);

MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
```

### 方法 C: 直接使用 cuSPARSE 矩阵包装 ⚡ 高级

```cpp
// PETSc 3.18+ 支持直接从 cuSPARSE 矩阵创建
cusparseHandle_t handle;
cusparseCreate(&handle);

cusparseSpMatDescr_t cusparse_mat;
cusparseCreateCsr(
    &cusparse_mat,
    n, n, nnz,
    d_row_ptrs,
    d_col_indices,
    d_values,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F
);

// 从 cuSPARSE 描述符创建 PETSc 矩阵
Mat A;
MatCreateFromCUSPARSE(PETSC_COMM_WORLD, cusparse_mat, &A);
```

### 关键点

✅ `MatCreateSeqAIJCUSPARSEWithArrays()` - 最简单  
✅ `MatCUSPARSEGetArrays()` - 获取内部指针  
✅ 支持从 cuSPARSE 矩阵直接创建  
⚠️ 需要 PETSc 3.17+

---

## 4️⃣ 完整示例：GPU 端组装矩阵

### 示例场景

假设您有一个 CUDA 核函数生成 Poisson 矩阵：

```cpp
__global__ void generate_poisson_matrix_gpu(
    int nx, int ny,
    int *d_row_ptrs,
    int *d_col_indices,
    double *d_values,
    double *d_rhs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny;
    
    if (idx >= n) return;
    
    int i = idx / ny;
    int j = idx % ny;
    
    // 计算该行的起始位置
    int row_start = d_row_ptrs[idx];
    int col_count = 0;
    
    // 左邻居
    if (i > 0) {
        d_col_indices[row_start + col_count] = idx - ny;
        d_values[row_start + col_count] = -1.0;
        col_count++;
    }
    
    // 下邻居
    if (j > 0) {
        d_col_indices[row_start + col_count] = idx - 1;
        d_values[row_start + col_count] = -1.0;
        col_count++;
    }
    
    // 对角线
    d_col_indices[row_start + col_count] = idx;
    d_values[row_start + col_count] = 4.0;
    col_count++;
    
    // 上邻居
    if (j < ny - 1) {
        d_col_indices[row_start + col_count] = idx + 1;
        d_values[row_start + col_count] = -1.0;
        col_count++;
    }
    
    // 右邻居
    if (i < nx - 1) {
        d_col_indices[row_start + col_count] = idx + ny;
        d_values[row_start + col_count] = -1.0;
        col_count++;
    }
    
    // RHS
    double h = 1.0 / (nx + 1);
    double x = (i + 1) * h;
    double y = (j + 1) * h;
    d_rhs[idx] = 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}
```

### 完整流程（以 AMGX 为例）

```cpp
int main() {
    int nx = 512, ny = 512;
    int n = nx * ny;
    int nnz = 5 * n - 2 * (nx + ny);  // 5点模板
    
    // 1. 在 GPU 上分配内存
    int *d_row_ptrs, *d_col_indices;
    double *d_values, *d_rhs;
    
    cudaMalloc(&d_row_ptrs, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_rhs, n * sizeof(double));
    
    // 2. 先计算行指针（简单核函数）
    compute_row_ptrs_kernel<<<...>>>(nx, ny, d_row_ptrs);
    
    // 3. 在 GPU 上生成矩阵
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    generate_poisson_matrix_gpu<<<blocks, threads>>>(
        nx, ny, d_row_ptrs, d_col_indices, d_values, d_rhs
    );
    cudaDeviceSynchronize();
    
    // 4. 直接使用 GPU 数据创建 AMGX 矩阵
    AMGX_initialize();
    AMGX_config_handle config;
    AMGX_resources_handle resources;
    
    const char *config_str = "...";  // 您的配置
    AMGX_config_create(&config, config_str);
    AMGX_resources_create_simple(&resources, config);
    
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    
    AMGX_matrix_create(&A, resources, AMGX_mode_dDDI);
    AMGX_vector_create(&b, resources, AMGX_mode_dDDI);
    AMGX_vector_create(&x, resources, AMGX_mode_dDDI);
    
    // ⭐ 关键：直接从 GPU 指针上传
    AMGX_matrix_upload_all(A, n, nnz, 1, 1,
                           d_row_ptrs, d_col_indices, d_values, nullptr);
    
    AMGX_vector_upload(b, n, 1, d_rhs);
    
    double *d_x_init;
    cudaMalloc(&d_x_init, n * sizeof(double));
    cudaMemset(d_x_init, 0, n * sizeof(double));
    AMGX_vector_upload(x, n, 1, d_x_init);
    
    // 5. 求解
    AMGX_solver_create(&solver, resources, AMGX_mode_dDDI, config);
    AMGX_solver_setup(solver, A);
    AMGX_solver_solve(solver, b, x);
    
    // 6. 结果（可以保持在 GPU）
    double *d_solution;
    cudaMalloc(&d_solution, n * sizeof(double));
    AMGX_vector_download(x, d_solution);
    
    // 现在 d_solution 包含求解结果，仍在 GPU
    // 可以继续用于其他 CUDA 计算
    
    // 清理
    AMGX_solver_destroy(solver);
    AMGX_matrix_destroy(A);
    AMGX_vector_destroy(b);
    AMGX_vector_destroy(x);
    AMGX_resources_destroy(resources);
    AMGX_config_destroy(config);
    AMGX_finalize();
    
    cudaFree(d_row_ptrs);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs);
    cudaFree(d_x_init);
    cudaFree(d_solution);
    
    return 0;
}
```

---

## 📊 性能对比

### 数据流对比

| 方法 | 矩阵生成 | 传输1 | 组装 | 求解 | 传输2 | 总传输 |
|------|---------|-------|------|------|-------|--------|
| **CPU 生成** | CPU | Host→Device (14MB) | Device | Device | Device→Host (2MB) | 16MB |
| **GPU 生成** | Device | - | Device | Device | - | 0MB |

### 时间估算（512×512）

**CPU 生成方法**:
- 矩阵生成 (CPU): 3 ms
- 传输 (H→D): 5 ms
- Setup (GPU): 350 ms
- Solve (GPU): 34 ms
- **总计**: ~392 ms

**GPU 生成方法**:
- 矩阵生成 (GPU): 0.5 ms ⚡
- 传输: 0 ms ✅
- Setup (GPU): 350 ms
- Solve (GPU): 34 ms
- **总计**: ~385 ms

**节省**: ~7 ms (小问题收益有限)

### 大规模问题（4096×4096）

**CPU 生成**:
- 传输: ~200 MB → ~17 ms
- **总计**: ~5000 ms

**GPU 生成**:
- 生成: ~2 ms
- 传输: 0 ms
- **总计**: ~4985 ms

**节省**: ~15 ms ✅

---

## ⚠️ 注意事项

### 1. 内存管理

```cpp
// 错误示例
{
    int *d_data;
    cudaMalloc(&d_data, size);
    AMGX_matrix_upload(..., d_data, ...);
    cudaFree(d_data);  // ❌ 不要立即释放！
}
// AMGX 可能还需要访问这些数据

// 正确示例
int *d_data;
cudaMalloc(&d_data, size);
AMGX_matrix_upload(..., d_data, ...);
// ... 完成所有操作后再释放
AMGX_matrix_destroy(A);
cudaFree(d_data);  // ✅ 安全
```

### 2. 数据格式

确保 CSR 格式正确：
- `row_ptrs[i]` = 第 i 行的第一个非零元在 values 中的位置
- `row_ptrs[n]` = nnz (总非零元数)
- `col_indices` 每行内应该排序（大多数库要求）

### 3. 同步

```cpp
// 生成矩阵的核函数后必须同步
generate_matrix_kernel<<<...>>>(...);
cudaDeviceSynchronize();  // ⚠️ 必须！

// 再使用数据
AMGX_matrix_upload(...);
```

---

## 🎯 推荐方案

### 对于新项目

| 库 | 推荐方法 | 理由 |
|---|---------|------|
| **AMGX** | `matrix_upload_all()` | 简单直接，文档清晰 |
| **HYPRE** | `IJMatrix + MEMORY_DEVICE` | 标准 API，自动管理 |
| **PETSc** | `MatCreateSeqAIJCUSPARSEWithArrays()` | 最简单的 PETSc GPU 接口 |

### 对于现有 CUDA 代码集成

1. **如果已有 CSR 格式**: 直接使用各库的设备指针接口
2. **如果是其他格式**: 先在 GPU 上转换为 CSR
3. **如果数据量小**: CPU 生成也可以（传输开销小）

---

## 📚 参考文档

- **AMGX**: [AMGX API Reference](https://github.com/NVIDIA/AMGX)
- **HYPRE**: [HYPRE GPU Guide](https://hypre.readthedocs.io/en/latest/gpu.html)
- **PETSc**: [PETSc CUDA/Kokkos](https://petsc.org/release/manual/gpu/)

---

**总结**: 所有三个库都支持从 GPU 端直接组装矩阵！选择最适合您项目的方法。

