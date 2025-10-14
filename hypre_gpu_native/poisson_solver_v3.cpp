#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "timer.h"
#include "matrix_generator.h"
#include "result_writer.h"
#include "config_reader.h"

int main(int argc, char** argv) {
    int myid, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (myid == 0) {
        std::cout << "===========================================\n";
        std::cout << "HYPRE GPU Benchmark (Working Version)\n";
        std::cout << "(CPU matrix gen + GPU solve)\n";
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
    std::string precond_type = config_reader.get_hypre_preconditioner(true);
    
    BenchmarkTimer timer;
    ResultWriter writer("../results");

    // ⚠️ 关键！在 HYPRE_Init 之前初始化 CUDA 设备
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
        std::cout << "Preconditioner: " << precond_type << "\n";
        std::cout << "Note: Matrix generated on CPU, solving on GPU\n\n";
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

        // 方案1：在 CPU 上生成矩阵（避免 CUDA 冲突）
        std::vector<int> rows, cols;
        std::vector<double> values, rhs_vec;
        
        if (myid == 0) {
            MatrixGenerator::generate_2d_poisson_5pt(nx, ny, rows, cols, values, rhs_vec);
            std::cout << "Matrix generated on CPU: " << n << " x " << n 
                      << ", nnz = " << values.size() << std::endl;
        }

        // 创建 IJMatrix（HYPRE 会自动分配 GPU 内存）
        HYPRE_BigInt ilower = 0, iupper = (myid == 0) ? n - 1 : -1;
        HYPRE_BigInt jlower = 0, jupper = (myid == 0) ? n - 1 : -1;

        HYPRE_IJMatrix A;
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, jlower, jupper, &A);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_DEVICE);

        // 从 CPU 数据填充矩阵
        if (myid == 0) {
            for (HYPRE_BigInt i = 0; i < n; i++) {
                HYPRE_Int row_start = rows[i];
                HYPRE_Int row_end = rows[i + 1];
                HYPRE_Int ncols = row_end - row_start;
                
                std::vector<HYPRE_Int> row_cols(ncols);
                std::vector<double> row_vals(ncols);
                
                for (HYPRE_Int j = 0; j < ncols; j++) {
                    row_cols[j] = cols[row_start + j];
                    row_vals[j] = values[row_start + j];
                }
                
                HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, row_cols.data(), row_vals.data());
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
            std::vector<HYPRE_BigInt> indices(n);
            for (HYPRE_BigInt i = 0; i < n; i++) indices[i] = i;
            
            HYPRE_IJVectorSetValues(b, n, indices.data(), rhs_vec.data());
            
            std::vector<double> zeros(n, 0.0);
            HYPRE_IJVectorSetValues(x, n, indices.data(), zeros.data());
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

