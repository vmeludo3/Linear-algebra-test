#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "timer.h"
#include "matrix_generator.h"
#include "result_writer.h"
#include "config_reader.h"

/**
 * HYPRE GPU Poisson 求解器示例
 * 使用 BoomerAMG 预条件器 + PCG 求解器 (GPU 模式)
 */

int main(int argc, char** argv) {
    // 初始化 MPI
    int myid, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (myid == 0) {
        std::cout << "===========================================\n";
        std::cout << "HYPRE GPU Poisson Solver Benchmark\n";
        std::cout << "===========================================\n\n";
        std::cout << "Running on " << num_procs << " MPI process(es)\n\n";
    }

    // 读取配置文件
    ConfigReader config_reader("../solver_config.yaml");
    if (myid == 0) {
        config_reader.print_summary("HYPRE-GPU");
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
        
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "GPU Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n\n";
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

    // 运行不同规模的测试
    for (int grid_size : grid_sizes) {
        if (myid == 0) {
            std::cout << "\n--- Testing with grid size: " << grid_size << "x" << grid_size << " ---\n";
        }
        
        int nx = grid_size;
        int ny = grid_size;
        int n = nx * ny;

        // 生成矩阵（仅在主进程）
        std::vector<int> rows, cols;
        std::vector<double> values, rhs_vec;
        
        if (myid == 0) {
            timer.start("matrix_generation");
            MatrixGenerator::generate_2d_poisson_5pt(nx, ny, rows, cols, values, rhs_vec);
            timer.stop("matrix_generation");
        }

        // 每个进程的局部大小
        int local_size = n / num_procs;
        int ilower = myid * local_size;
        int iupper = (myid + 1) * local_size - 1;
        if (myid == num_procs - 1) {
            iupper = n - 1;
            local_size = n - ilower;
        }

        if (myid == 0) {
            std::cout << "Problem size: " << n << " x " << n << std::endl;
        }

        timer.start("setup");

        // 创建 HYPRE IJMatrix
        HYPRE_IJMatrix A;
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        // 填充矩阵
        if (myid == 0) {
            // 转换为 CSR 格式并插入
            std::vector<int> row_ptrs(n + 1, 0);
            for (int r : rows) {
                row_ptrs[r + 1]++;
            }
            for (int i = 0; i < n; i++) {
                row_ptrs[i + 1] += row_ptrs[i];
            }

            for (int i = ilower; i <= iupper; i++) {
                int nnz_row = row_ptrs[i + 1] - row_ptrs[i];
                std::vector<int> row_cols(nnz_row);
                std::vector<double> row_vals(nnz_row);
                
                for (int j = 0; j < nnz_row; j++) {
                    int idx = row_ptrs[i] + j;
                    row_cols[j] = cols[idx];
                    row_vals[j] = values[idx];
                }
                
                HYPRE_IJMatrixSetValues(A, 1, &nnz_row, &i, row_cols.data(), row_vals.data());
            }
        }

        HYPRE_IJMatrixAssemble(A);
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);

        // 创建向量
        HYPRE_IJVector b, x;
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
        HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b);

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
        HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x);

        // 设置向量值
        if (myid == 0) {
            std::vector<int> indices(local_size);
            std::vector<double> rhs_local(local_size);
            std::vector<double> x_local(local_size, 0.0);
            
            for (int i = 0; i < local_size; i++) {
                indices[i] = ilower + i;
                rhs_local[i] = rhs_vec[ilower + i];
            }
            
            HYPRE_IJVectorSetValues(b, local_size, indices.data(), rhs_local.data());
            HYPRE_IJVectorSetValues(x, local_size, indices.data(), x_local.data());
        }

        HYPRE_IJVectorAssemble(b);
        HYPRE_IJVectorAssemble(x);

        HYPRE_ParVector par_b, par_x;
        HYPRE_IJVectorGetObject(b, (void **) &par_b);
        HYPRE_IJVectorGetObject(x, (void **) &par_x);

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
            // 使用 BoomerAMG 预条件器
            HYPRE_BoomerAMGCreate(&precond);
            HYPRE_BoomerAMGSetPrintLevel(precond, 0);
            HYPRE_BoomerAMGSetCoarsenType(precond, 6);  // Falgout coarsening
            HYPRE_BoomerAMGSetRelaxType(precond, 6);    // Symmetric G-S
            HYPRE_BoomerAMGSetNumSweeps(precond, 1);
            HYPRE_BoomerAMGSetMaxLevels(precond, 20);
            HYPRE_BoomerAMGSetTol(precond, 0.0);
            HYPRE_BoomerAMGSetMaxIter(precond, 1);
            
            HYPRE_PCGSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                               precond);
        }
        // else: 无预条件器，PCG 直接求解

        HYPRE_PCGSetup(solver, (HYPRE_Matrix)parcsr_A, 
                      (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);
        
        timer.stop("setup");

        // 求解
        timer.start("solve");
        HYPRE_PCGSolve(solver, (HYPRE_Matrix)parcsr_A, 
                      (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);
        
        // 同步 GPU
        cudaDeviceSynchronize();
        timer.stop("solve");

        // 获取求解信息
        int num_iterations;
        double final_res_norm;
        HYPRE_PCGGetNumIterations(solver, &num_iterations);
        HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

        if (myid == 0) {
            // 保存结果
            ResultWriter::BenchmarkResult result;
            result.library_name = "HYPRE-GPU";
            result.test_name = "Poisson_2D_PCG_BoomerAMG_GPU";
            result.problem_size = n;
            result.setup_time = timer.get_total_time("setup");
            result.solve_time = timer.get_total_time("solve");
            result.total_time = result.setup_time + result.solve_time;
            result.iterations = num_iterations;
            result.residual = final_res_norm;
            
            writer.add_result(result);

            std::cout << "Iterations: " << num_iterations << std::endl;
            std::cout << "Final residual: " << final_res_norm << std::endl;
            std::cout << "Setup time: " << result.setup_time << " s" << std::endl;
            std::cout << "Solve time: " << result.solve_time << " s" << std::endl;
            std::cout << "Total time: " << result.total_time << " s" << std::endl;
        }

        // 清理
        if (use_precond) {
            HYPRE_BoomerAMGDestroy(precond);
        }
        HYPRE_ParCSRPCGDestroy(solver);
        HYPRE_IJMatrixDestroy(A);
        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);

        timer.reset("setup");
        timer.reset("solve");
    }

    if (myid == 0) {
        std::cout << "\n";
        timer.print_summary();
        writer.print_summary();
        writer.write_csv("hypre_gpu_results.csv");
        writer.write_json("hypre_gpu_results.json");
        std::cout << "\nHYPRE GPU benchmark completed!\n";
    }

    HYPRE_Finalize();

    MPI_Finalize();
    return 0;
}

