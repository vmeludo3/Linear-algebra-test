#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <petsc.h>
#include "timer.h"
#include "result_writer.h"
#include "config_reader.h"
#include "gpu_matrix_generator.cuh"

static char help[] = "PETSc GPU Native Poisson Solver (Matrix assembled on GPU).\n\n";

int main(int argc, char** argv) {
    PetscErrorCode ierr;
    PetscMPIInt myid, num_procs;

    // 设置环境变量禁用 GPU-aware MPI 检查（WSL 环境需要）
    setenv("PETSC_OPTIONS", "-use_gpu_aware_mpi 0", 0);
    
    ierr = PetscInitialize(&argc, &argv, nullptr, help); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &myid); CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &num_procs); CHKERRQ(ierr);

    if (myid == 0) {
        std::cout << "===========================================\n";
        std::cout << "PETSc GPU Native Poisson Solver Benchmark\n";
        std::cout << "(Matrix assembled directly on GPU)\n";
        std::cout << "===========================================\n\n";
        std::cout << "Running on " << num_procs << " MPI process(es)\n\n";
    }

    // 读取配置文件
    ConfigReader config_reader("../solver_config.yaml");
    if (myid == 0) {
        config_reader.print_summary("PETSc-GPU-Native");
    }
    
    // 从配置文件获取参数
    std::vector<int> grid_sizes = config_reader.get_grid_sizes();
    std::string precond_type = config_reader.get_petsc_preconditioner(true); // true = GPU
    
    BenchmarkTimer timer;
    ResultWriter writer("../results");
    
    if (myid == 0) {
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

        // 创建矩阵（CUDA 类型）
        Mat A;
        ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
        
        PetscInt local_size = (myid == 0) ? n : 0;
        ierr = MatSetSizes(A, local_size, local_size, n, n); CHKERRQ(ierr);
        ierr = MatSetType(A, MATAIJCUSPARSE); CHKERRQ(ierr);  // ← CUDA 矩阵
        ierr = MatSetUp(A); CHKERRQ(ierr);

        // 从 GPU 数据填充矩阵
        if (myid == 0) {
            // 将 GPU 数据拷贝到 CPU（PETSc API 需要）
            std::vector<int> h_row_ptrs(n+1), h_col_indices(nnz);
            std::vector<double> h_values(nnz);
            
            cudaMemcpy(h_row_ptrs.data(), d_row_ptrs, (n+1) * sizeof(int), 
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_col_indices.data(), d_col_indices, nnz * sizeof(int), 
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_values.data(), d_values, nnz * sizeof(double), 
                       cudaMemcpyDeviceToHost);

            for (PetscInt i = 0; i < n; i++) {
                PetscInt ncols = h_row_ptrs[i+1] - h_row_ptrs[i];
                std::vector<PetscInt> cols(ncols);
                std::vector<PetscScalar> vals(ncols);
                
                for (PetscInt j = 0; j < ncols; j++) {
                    cols[j] = h_col_indices[h_row_ptrs[i] + j];
                    vals[j] = h_values[h_row_ptrs[i] + j];
                }
                
                ierr = MatSetValues(A, 1, &i, ncols, cols.data(), vals.data(), 
                                   INSERT_VALUES); CHKERRQ(ierr);
            }
        }

        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        // 创建向量（CUDA 类型）
        Vec b, x;
        ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
        ierr = VecSetSizes(b, local_size, n); CHKERRQ(ierr);
        ierr = VecSetType(b, VECCUDA); CHKERRQ(ierr);  // ← CUDA 向量
        ierr = VecSetUp(b); CHKERRQ(ierr);

        ierr = VecDuplicate(b, &x); CHKERRQ(ierr);

        // 设置 RHS
        if (myid == 0) {
            std::vector<double> h_rhs(n);
            cudaMemcpy(h_rhs.data(), d_rhs, n * sizeof(double), 
                       cudaMemcpyDeviceToHost);
            
            for (PetscInt i = 0; i < n; i++) {
                PetscScalar val = h_rhs[i];
                ierr = VecSetValues(b, 1, &i, &val, INSERT_VALUES); CHKERRQ(ierr);
            }
        }

        ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
        ierr = VecSet(x, 0.0); CHKERRQ(ierr);

        // 创建 KSP 求解器
        KSP ksp;
        ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
        
        // 设置求解器类型为 CG
        ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
        
        // 根据配置设置预条件器
        PC pc;
        ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
        
        if (precond_type == "GAMG") {
            ierr = PCSetType(pc, PCGAMG); CHKERRQ(ierr);
            ierr = PCGAMGSetNSmooths(pc, 1); CHKERRQ(ierr);
            ierr = PCGAMGSetType(pc, PCGAMGAGG); CHKERRQ(ierr);
        } else if (precond_type == "JACOBI") {
            ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
        } else {
            ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
        }
        
        // 设置收敛参数
        ierr = KSPSetTolerances(ksp, 1e-6, 1e-8, PETSC_DEFAULT, 2000); CHKERRQ(ierr);
        ierr = KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED); CHKERRQ(ierr);
        ierr = KSPConvergedDefaultSetUIRNorm(ksp); CHKERRQ(ierr);
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
        
        ierr = KSPSetUp(ksp); CHKERRQ(ierr);
        
        timer.stop("setup");

        // Solve
        timer.start("solve");
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
        timer.stop("solve");
        timer.stop("total");

        // 获取求解信息
        PetscInt num_iterations;
        PetscReal residual_norm, rnorm0;
        KSPConvergedReason reason;
        
        ierr = KSPGetIterationNumber(ksp, &num_iterations); CHKERRQ(ierr);
        ierr = KSPGetResidualNorm(ksp, &residual_norm); CHKERRQ(ierr);
        ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
        
        // 计算相对残差
        Vec residual;
        ierr = VecDuplicate(b, &residual); CHKERRQ(ierr);
        ierr = MatMult(A, x, residual); CHKERRQ(ierr);
        ierr = VecAYPX(residual, -1.0, b); CHKERRQ(ierr);
        ierr = VecNorm(residual, NORM_2, &residual_norm); CHKERRQ(ierr);
        ierr = VecNorm(b, NORM_2, &rnorm0); CHKERRQ(ierr);
        PetscReal relative_residual = residual_norm / rnorm0;
        ierr = VecDestroy(&residual); CHKERRQ(ierr);

        if (myid == 0) {
            ResultWriter::BenchmarkResult result;
            result.library_name = "PETSc-GPU-Native";
            result.test_name = "Poisson_2D_CG_" + precond_type + "_GPUNative";
            result.problem_size = n;
            result.setup_time = timer.get_total_time("setup");
            result.solve_time = timer.get_total_time("solve");
            result.total_time = timer.get_total_time("total");
            result.iterations = num_iterations;
            result.residual = relative_residual;
            
            writer.add_result(result);

            std::cout << "Iterations: " << num_iterations << std::endl;
            std::cout << "Absolute residual: " << residual_norm << std::endl;
            std::cout << "Relative residual: " << relative_residual << std::endl;
            std::cout << "Converged: " << (reason > 0 ? "Yes" : "No") << std::endl;
            std::cout << "Setup time: " << result.setup_time << " s" << std::endl;
            std::cout << "Solve time: " << result.solve_time << " s" << std::endl;
            std::cout << "Total time: " << result.total_time << " s" << std::endl;
            std::cout << std::endl;
        }

        // 清理
        ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
        ierr = MatDestroy(&A); CHKERRQ(ierr);
        ierr = VecDestroy(&b); CHKERRQ(ierr);
        ierr = VecDestroy(&x); CHKERRQ(ierr);
        
        if (myid == 0) {
            cudaFree(d_row_ptrs);
            cudaFree(d_col_indices);
            cudaFree(d_values);
            cudaFree(d_rhs);
        }
    }

    if (myid == 0) {
        std::cout << "Saving results..." << std::endl;
        writer.write_csv("petsc_gpu_native_results.csv");
        writer.write_json("petsc_gpu_native_results.json");
        std::cout << "Results saved to ../results/" << std::endl;
    }

    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}

