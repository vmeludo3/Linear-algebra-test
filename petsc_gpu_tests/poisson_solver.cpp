#include <iostream>
#include <vector>
#include <string>
#include <petsc.h>
#include "timer.h"
#include "matrix_generator.h"
#include "result_writer.h"
#include "config_reader.h"

/**
 * PETSc GPU Poisson 求解器示例
 * 使用 GAMG 预条件器 + CG 求解器 (GPU 模式)
 */

static char help[] = "PETSc GPU Poisson solver benchmark\n\n";

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
        std::cout << "PETSc GPU Poisson Solver Benchmark\n";
        std::cout << "===========================================\n\n";
        std::cout << "Running on " << num_procs << " MPI process(es)\n\n";
    }

    // 读取配置文件
    ConfigReader config_reader("../solver_config.yaml");
    if (myid == 0) {
        config_reader.print_summary("PETSc-GPU");
    }
    
    // 从配置文件获取参数
    std::vector<int> grid_sizes = config_reader.get_grid_sizes();
    std::string precond_type = config_reader.get_petsc_preconditioner(true); // true = GPU
    
    BenchmarkTimer timer;
    ResultWriter writer("../results");
    
    if (myid == 0) {
        std::cout << "Preconditioner: " << precond_type << "\n\n";
    }

    // 运行不同规模的测试
    for (int grid_size : grid_sizes) {
        if (myid == 0) {
            std::cout << "\n--- Testing with grid size: " << grid_size << "x" << grid_size << " ---\n";
        }
        
        PetscInt nx = grid_size;
        PetscInt ny = grid_size;
        PetscInt n = nx * ny;

        // 生成矩阵（仅在主进程）
        std::vector<int> rows_vec, cols_vec;
        std::vector<double> values_vec, rhs_vec;
        
        if (myid == 0) {
            timer.start("matrix_generation");
            MatrixGenerator::generate_2d_poisson_5pt(nx, ny, rows_vec, cols_vec, values_vec, rhs_vec);
            timer.stop("matrix_generation");
        }

        // 每个进程的局部大小
        PetscInt local_size = n / num_procs;
        PetscInt ilower = myid * local_size;
        PetscInt iupper = (myid + 1) * local_size;
        if (myid == num_procs - 1) {
            iupper = n;
        }
        local_size = iupper - ilower;

        if (myid == 0) {
            std::cout << "Problem size: " << n << " x " << n << std::endl;
        }

        timer.start("setup");

        // 创建 PETSc 矩阵 - 使用 CUDA 类型
        Mat A;
        ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
        ierr = MatSetSizes(A, local_size, local_size, n, n); CHKERRQ(ierr);
        ierr = MatSetType(A, MATAIJCUSPARSE); CHKERRQ(ierr);  // 使用 CUDA 稀疏矩阵
        ierr = MatSetUp(A); CHKERRQ(ierr);

        // 填充矩阵
        if (myid == 0) {
            // 转换为 CSR 格式
            std::vector<PetscInt> row_ptrs(n + 1, 0);
            for (int r : rows_vec) {
                row_ptrs[r + 1]++;
            }
            for (PetscInt i = 0; i < n; i++) {
                row_ptrs[i + 1] += row_ptrs[i];
            }

            for (PetscInt i = ilower; i < iupper; i++) {
                PetscInt nnz_row = row_ptrs[i + 1] - row_ptrs[i];
                std::vector<PetscInt> row_cols(nnz_row);
                std::vector<PetscScalar> row_vals(nnz_row);
                
                for (PetscInt j = 0; j < nnz_row; j++) {
                    PetscInt idx = row_ptrs[i] + j;
                    row_cols[j] = cols_vec[idx];
                    row_vals[j] = values_vec[idx];
                }
                
                ierr = MatSetValues(A, 1, &i, nnz_row, row_cols.data(), 
                                   row_vals.data(), INSERT_VALUES); CHKERRQ(ierr);
            }
        }

        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        // 创建向量 - 使用 CUDA 类型
        Vec b, x;
        ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
        ierr = VecSetSizes(b, local_size, n); CHKERRQ(ierr);
        ierr = VecSetType(b, VECCUDA); CHKERRQ(ierr);  // 使用 CUDA 向量

        ierr = VecDuplicate(b, &x); CHKERRQ(ierr);

        // 设置向量值
        if (myid == 0) {
            for (PetscInt i = ilower; i < iupper; i++) {
                PetscScalar val = rhs_vec[i];
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
            // 使用 GAMG 预条件器
            ierr = PCSetType(pc, PCGAMG); CHKERRQ(ierr);
            ierr = PCGAMGSetNSmooths(pc, 1); CHKERRQ(ierr);
            ierr = PCGAMGSetType(pc, PCGAMGAGG); CHKERRQ(ierr);
        } else if (precond_type == "JACOBI") {
            // 使用 Jacobi 预条件器
            ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
        } else {
            // 无预条件器
            ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
        }
        
        // 设置收敛参数：相对容差 1e-6，绝对容差 1e-8，最大迭代 2000
        ierr = KSPSetTolerances(ksp, 1e-6, 1e-8, PETSC_DEFAULT, 2000); CHKERRQ(ierr);
        
        // 设置残差范数类型为真实残差（不是预条件残差）
        ierr = KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED); CHKERRQ(ierr);
        
        // 设置初始残差作为参考
        ierr = KSPConvergedDefaultSetUIRNorm(ksp); CHKERRQ(ierr);
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
        
        ierr = KSPSetUp(ksp); CHKERRQ(ierr);
        
        timer.stop("setup");

        // 求解
        timer.start("solve");
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
        timer.stop("solve");

        // 获取求解信息
        PetscInt num_iterations;
        PetscReal residual_norm, rnorm0;
        ierr = KSPGetIterationNumber(ksp, &num_iterations); CHKERRQ(ierr);
        ierr = KSPGetResidualNorm(ksp, &residual_norm); CHKERRQ(ierr);
        
        // 计算相对残差（与初始残差相比）
        Vec residual;
        ierr = VecDuplicate(b, &residual); CHKERRQ(ierr);
        ierr = MatMult(A, x, residual); CHKERRQ(ierr);
        ierr = VecAYPX(residual, -1.0, b); CHKERRQ(ierr);  // residual = b - A*x
        ierr = VecNorm(residual, NORM_2, &residual_norm); CHKERRQ(ierr);
        ierr = VecNorm(b, NORM_2, &rnorm0); CHKERRQ(ierr);
        PetscReal relative_residual = residual_norm / rnorm0;
        ierr = VecDestroy(&residual); CHKERRQ(ierr);

        // 检查收敛
        KSPConvergedReason reason;
        ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);

        if (myid == 0) {
            // 保存结果
            ResultWriter::BenchmarkResult result;
            result.library_name = "PETSc-GPU";
            result.test_name = "Poisson_2D_CG_GAMG_GPU";
            result.problem_size = n;
            result.setup_time = timer.get_total_time("setup");
            result.solve_time = timer.get_total_time("solve");
            result.total_time = result.setup_time + result.solve_time;
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
        }

        // 清理
        ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
        ierr = MatDestroy(&A); CHKERRQ(ierr);
        ierr = VecDestroy(&b); CHKERRQ(ierr);
        ierr = VecDestroy(&x); CHKERRQ(ierr);

        timer.reset("setup");
        timer.reset("solve");
    }

    if (myid == 0) {
        std::cout << "\n";
        timer.print_summary();
        writer.print_summary();
        writer.write_csv("petsc_gpu_results.csv");
        writer.write_json("petsc_gpu_results.json");
        std::cout << "\nPETSc GPU benchmark completed!\n";
    }

    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}

