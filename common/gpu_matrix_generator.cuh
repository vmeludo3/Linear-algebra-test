#ifndef GPU_MATRIX_GENERATOR_CUH
#define GPU_MATRIX_GENERATOR_CUH

#include <cuda_runtime.h>
#include <cmath>

// CUDA 核函数：计算 CSR 行指针
__global__ void compute_row_ptrs_kernel(int nx, int ny, int* d_row_ptrs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny;
    
    if (idx > n) return;  // 注意：需要 n+1 个元素
    
    if (idx == 0) {
        d_row_ptrs[0] = 0;
        return;
    }
    
    if (idx == n) {
        // 最后一个元素：总非零元数
        int nnz = 5 * n - 2 * (nx + ny);
        d_row_ptrs[n] = nnz;
        return;
    }
    
    // 计算每行的非零元个数
    int i = (idx - 1) / ny;
    int j = (idx - 1) % ny;
    int count = 1;  // 对角线
    
    if (i > 0) count++;      // 左邻居
    if (j > 0) count++;      // 下邻居
    if (j < ny - 1) count++; // 上邻居
    if (i < nx - 1) count++; // 右邻居
    
    // 累加前面所有行的非零元数
    int prev_count = 0;
    for (int k = 0; k < idx - 1; k++) {
        int ki = k / ny;
        int kj = k % ny;
        int kcount = 1;
        if (ki > 0) kcount++;
        if (kj > 0) kcount++;
        if (kj < ny - 1) kcount++;
        if (ki < nx - 1) kcount++;
        prev_count += kcount;
    }
    
    d_row_ptrs[idx] = prev_count;
}

// 更高效的行指针计算（使用前缀和）
__global__ void compute_row_ptrs_efficient_kernel(int nx, int ny, int* d_row_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny;
    
    if (idx >= n) return;
    
    int i = idx / ny;
    int j = idx % ny;
    int count = 1;  // 对角线
    
    if (i > 0) count++;      // 左邻居
    if (j > 0) count++;      // 下邻居
    if (j < ny - 1) count++; // 上邻居
    if (i < nx - 1) count++; // 右邻居
    
    d_row_counts[idx] = count;
}

// CUDA 核函数：生成 Poisson 矩阵（5点模板）
__global__ void generate_poisson_matrix_kernel(
    int nx, int ny,
    const int* d_row_ptrs,
    int* d_col_indices,
    double* d_values,
    double* d_rhs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny;
    
    if (idx >= n) return;
    
    int i = idx / ny;
    int j = idx % ny;
    
    int row_start = d_row_ptrs[idx];
    int col = 0;
    
    // 左邻居
    if (i > 0) {
        d_col_indices[row_start + col] = idx - ny;
        d_values[row_start + col] = -1.0;
        col++;
    }
    
    // 下邻居
    if (j > 0) {
        d_col_indices[row_start + col] = idx - 1;
        d_values[row_start + col] = -1.0;
        col++;
    }
    
    // 对角线
    d_col_indices[row_start + col] = idx;
    d_values[row_start + col] = 4.0;
    col++;
    
    // 上邻居
    if (j < ny - 1) {
        d_col_indices[row_start + col] = idx + 1;
        d_values[row_start + col] = -1.0;
        col++;
    }
    
    // 右邻居
    if (i < nx - 1) {
        d_col_indices[row_start + col] = idx + ny;
        d_values[row_start + col] = -1.0;
        col++;
    }
    
    // RHS: f(x,y) = 2π² sin(πx) sin(πy)
    double h = 1.0 / (nx + 1);
    double x = (i + 1) * h;
    double y = (j + 1) * h;
    d_rhs[idx] = 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

// 辅助函数：在主机端调用
inline void generate_poisson_matrix_gpu(
    int nx, int ny,
    int** d_row_ptrs_out,
    int** d_col_indices_out,
    double** d_values_out,
    double** d_rhs_out,
    int* nnz_out
) {
    int n = nx * ny;
    int nnz = 5 * n - 2 * (nx + ny);
    
    // 分配设备内存
    int *d_row_ptrs, *d_col_indices, *d_row_counts;
    double *d_values, *d_rhs;
    
    cudaMalloc(&d_row_ptrs, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_rhs, n * sizeof(double));
    cudaMalloc(&d_row_counts, n * sizeof(int));
    
    // 1. 计算每行的非零元个数
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    compute_row_ptrs_efficient_kernel<<<blocks, threads>>>(nx, ny, d_row_counts);
    
    // 2. 使用 thrust 计算前缀和得到行指针
    // 为简单起见，我们用简单的核函数
    cudaMemset(d_row_ptrs, 0, sizeof(int));  // d_row_ptrs[0] = 0
    
    // CPU 端计算行指针（更简单，开销小）
    int *h_row_ptrs = new int[n + 1];
    h_row_ptrs[0] = 0;
    for (int i = 0; i < n; i++) {
        int ii = i / ny;
        int jj = i % ny;
        int count = 1;
        if (ii > 0) count++;
        if (jj > 0) count++;
        if (jj < ny - 1) count++;
        if (ii < nx - 1) count++;
        h_row_ptrs[i + 1] = h_row_ptrs[i] + count;
    }
    cudaMemcpy(d_row_ptrs, h_row_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_row_ptrs;
    
    // 3. 生成矩阵
    generate_poisson_matrix_kernel<<<blocks, threads>>>(
        nx, ny, d_row_ptrs, d_col_indices, d_values, d_rhs
    );
    
    cudaDeviceSynchronize();
    cudaFree(d_row_counts);
    
    // 返回指针
    *d_row_ptrs_out = d_row_ptrs;
    *d_col_indices_out = d_col_indices;
    *d_values_out = d_values;
    *d_rhs_out = d_rhs;
    *nnz_out = nnz;
}

#endif // GPU_MATRIX_GENERATOR_CUH

