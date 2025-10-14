#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "timer.h"
#include "result_writer.h"
#include "config_reader.h"
#include "gpu_matrix_generator.cuh"

int main(int argc, char** argv) {
    int myid, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (myid == 0) {
        std::cout << "===========================================\n";
        std::cout << "HYPRE GPU Native Poisson Solver Benchmark\n";
        std::cout << "(Matrix assembled directly on GPU)\n";
        std::cout << "===========================================\n\n";
        std::cout << "Running on " << num_procs << " MPI process(es)\n\n";
    }

    // 读取配置文件
    ConfigReader config_reader("../solver_config.yaml");
    if (myid == 0) {
        config_reader.print_summary("HYPRE-GPU-Native");
    }
    
    // 从配置文件获取参数
    std::vector<int> grid_sizes = config_reader.get_grid_sizes();
    std::string precond_type = config_reader.get_hypre_preconditioner(true); // true = GPU
    
    BenchmarkTimer timer;
    ResultWriter writer("../results");

    // 初始化 CUDA 设备（关键！在 HYPRE_Init 之前）
    if (myid == 0) {
        cudaSetDevice(0);
        cudaFree(0);  // 强制创建 CUDA 上下文
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "Using GPU: " << prop.name << "\n";
            std::cout << "GPU Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n";
        } else {
            std::cout << "No CUDA devices found. Test will likely fail.\n\n";
        }
    }

    // 初始化 HYPRE
    HYPRE_Init();
    
    // 设置 HYPRE 使用 GPU 内存和执行
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

    if (myid == 0) {
        std::cout << "HYPRE configured for GPU execution\n";
        std::cout << "Preconditioner: " << precond_type << "\n\n";
    }

    // 测试不同网格尺寸
    for (int grid_size : grid_sizes) {
        int nx = grid_size, ny = grid_size;
        int n = nx * ny;
        
        if (myid == 0) {
            std::cout << "Testing grid size: " << nx << " x " << ny 
                      << " (" << n << " unknowns)" << std::endl;
        }

        timer.reset();
        timer.start("total");
        timer.start("setup");

        // ⭐ 关键：在 GPU 上生成矩阵（仅在 rank 0）
        int *d_row_ptrs = nullptr, *d_col_indices = nullptr;
        double *d_values = nullptr, *d_rhs = nullptr;
        int nnz = 0;
        
        if (myid == 0) {
            generate_poisson_matrix_gpu(nx, ny, &d_row_ptrs, &d_col_indices, 
                                       &d_values, &d_rhs, &nnz);
            
            std::cout << "Matrix generated on GPU: " << n << " x " << n 
                      << ", nnz = " << nnz << std::endl;
        }

        // 创建 IJMatrix（HYPRE 会自动分配 GPU 内存）
        HYPRE_BigInt ilower = 0, iupper = (myid == 0) ? n - 1 : -1;
        HYPRE_BigInt jlower = 0, jupper = (myid == 0) ? n - 1 : -1;

        HYPRE_IJMatrix A;
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, jlower, jupper, &A);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_DEVICE);

        // 从 GPU 数据填充矩阵
        if (myid == 0) {
            // 将 GPU 数据拷贝到临时 CPU 缓冲区
            // （HYPRE API 目前需要通过 host 接口）
            std::vector<int> h_row_ptrs(n+1), h_col_indices(nnz);
            std::vector<double> h_values(nnz);
            
            cudaMemcpy(h_row_ptrs.data(), d_row_ptrs, (n+1) * sizeof(int), 
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_col_indices.data(), d_col_indices, nnz * sizeof(int), 
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_values.data(), d_values, nnz * sizeof(double), 
                       cudaMemcpyDeviceToHost);

            for (HYPRE_BigInt i = 0; i < n; i++) {
                HYPRE_Int ncols = h_row_ptrs[i+1] - h_row_ptrs[i];
                HYPRE_Int *cols = &h_col_indices[h_row_ptrs[i]];
                double *vals = &h_values[h_row_ptrs[i]];
                
                HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, cols, vals);
            }
        }

        HYPRE_IJMatrixAssemble(A);

        // 创建向量
        HYPRE_IJVector b, x;
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
        
        HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
        HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
        
        HYPRE_IJVectorInitialize_v2(b, HYPRE_MEMORY_DEVICE);
        HYPRE_IJVectorInitialize_v2(x, HYPRE_MEMORY_DEVICE);

        // 设置 RHS
        if (myid == 0) {
            std::vector<double> h_rhs(n);
            cudaMemcpy(h_rhs.data(), d_rhs, n * sizeof(double), 
                       cudaMemcpyDeviceToHost);
            
            std::vector<HYPRE_BigInt> indices(n);
            for (HYPRE_BigInt i = 0; i < n; i++) indices[i] = i;
            
            HYPRE_IJVectorSetValues(b, n, indices.data(), h_rhs.data());
            HYPRE_IJVectorSetValues(x, n, indices.data(), nullptr);  // 初始化为 0
        }

        HYPRE_IJVectorAssemble(b);
        HYPRE_IJVectorAssemble(x);

        // 获取 ParCSR 对象
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_ParVector par_b, par_x;
        
        HYPRE_IJMatrixGetObject(A, (void**)&parcsr_A);
        HYPRE_IJVectorGetObject(b, (void**)&par_b);
        HYPRE_IJVectorGetObject(x, (void**)&par_x);

        // 创建 PCG 求解器
        HYPRE_Solver solver;
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
        HYPRE_PCGSetMaxIter(solver, 1000);
        HYPRE_PCGSetTol(solver, 1e-6);
        HYPRE_PCGSetTwoNorm(solver, 1);
        HYPRE_PCGSetPrintLevel(solver, 0);
        
        // 根据配置创建预条件器
        HYPRE_Solver precond = nullptr;
        bool use_precond = (precond_type == "BOOMERAMG");

        if (use_precond) {
            HYPRE_BoomerAMGCreate(&precond);
            HYPRE_BoomerAMGSetPrintLevel(precond, 0);
            HYPRE_BoomerAMGSetCoarsenType(precond, 6);
            HYPRE_BoomerAMGSetRelaxType(precond, 6);
            HYPRE_BoomerAMGSetNumSweeps(precond, 1);
            HYPRE_BoomerAMGSetMaxLevels(precond, 20);
            HYPRE_BoomerAMGSetTol(precond, 0.0);
            HYPRE_BoomerAMGSetMaxIter(precond, 1);
            
            HYPRE_PCGSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                               precond);
        }

        HYPRE_PCGSetup(solver, (HYPRE_Matrix)parcsr_A, 
                      (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);
        
        timer.stop("setup");

        // Solve
        timer.start("solve");
        HYPRE_PCGSolve(solver, (HYPRE_Matrix)parcsr_A, 
                      (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);
        cudaDeviceSynchronize();
        timer.stop("solve");
        timer.stop("total");

        // 获取求解信息
        HYPRE_Int num_iterations;
        double final_res_norm;
        HYPRE_PCGGetNumIterations(solver, &num_iterations);
        HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

        if (myid == 0) {
            ResultWriter::BenchmarkResult result;
            result.library_name = "HYPRE-GPU-Native";
            result.test_name = "Poisson_2D_PCG_" + precond_type + "_GPUNative";
            result.problem_size = n;
            result.setup_time = timer.get_total_time("setup");
            result.solve_time = timer.get_total_time("solve");
            result.total_time = timer.get_total_time("total");
            result.iterations = num_iterations;
            result.residual = final_res_norm;
            
            writer.add_result(result);

            std::cout << "Iterations: " << num_iterations << std::endl;
            std::cout << "Relative residual: " << final_res_norm << std::endl;
            std::cout << "Setup time: " << result.setup_time << " s" << std::endl;
            std::cout << "Solve time: " << result.solve_time << " s" << std::endl;
            std::cout << "Total time: " << result.total_time << " s" << std::endl;
            std::cout << std::endl;
        }

        // 清理
        if (use_precond) {
            HYPRE_BoomerAMGDestroy(precond);
        }
        HYPRE_ParCSRPCGDestroy(solver);
        HYPRE_IJMatrixDestroy(A);
        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);
        
        if (myid == 0) {
            cudaFree(d_row_ptrs);
            cudaFree(d_col_indices);
            cudaFree(d_values);
            cudaFree(d_rhs);
        }
    }

    if (myid == 0) {
        std::cout << "Saving results..." << std::endl;
        writer.write_csv("hypre_gpu_native_results.csv");
        writer.write_json("hypre_gpu_native_results.json");
        std::cout << "Results saved to ../results/" << std::endl;
    }

    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}

