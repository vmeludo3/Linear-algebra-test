#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "amgx_c.h"
#include "timer.h"
#include "result_writer.h"
#include "config_reader.h"
#include "gpu_matrix_generator.cuh"

void check_amgx_error(AMGX_RC err, const char* msg) {
    if (err != AMGX_RC_OK) {
        std::cerr << "AMGX Error: " << msg << " (code " << err << ")" << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    std::cout << "===========================================\n";
    std::cout << "AMGX GPU Native Poisson Solver Benchmark\n";
    std::cout << "(Matrix assembled directly on GPU)\n";
    std::cout << "===========================================\n\n";

    // 读取配置文件
    ConfigReader config_reader("../solver_config.yaml");
    config_reader.print_summary("AMGX-GPU-Native");
    
    // 从配置文件获取参数
    std::vector<int> grid_sizes = config_reader.get_grid_sizes();
    std::string precond_type = config_reader.get_amgx_preconditioner();
    
    BenchmarkTimer timer;
    ResultWriter writer("../results");

    // 初始化 AMGX
    check_amgx_error(AMGX_initialize(), "AMGX initialization");
    check_amgx_error(AMGX_initialize_plugins(), "AMGX plugins initialization");

    // 根据配置文件生成 AMGX 配置字符串
    std::string config_str;
    
    if (precond_type == "AMG") {
        config_str = 
            "{\n"
            "  \"config_version\": 2,\n"
            "  \"solver\": {\n"
            "    \"solver\": \"PCG\",\n"
            "    \"preconditioner\": {\n"
            "      \"solver\": \"AMG\",\n"
            "      \"algorithm\": \"AGGREGATION\",\n"
            "      \"selector\": \"SIZE_2\",\n"
            "      \"smoother\": \"BLOCK_JACOBI\",\n"
            "      \"presweeps\": 1,\n"
            "      \"postsweeps\": 1,\n"
            "      \"max_iters\": 1,\n"
            "      \"cycle\": \"V\"\n"
            "    },\n"
            "    \"print_solve_stats\": 1,\n"
            "    \"monitor_residual\": 1,\n"
            "    \"max_iters\": 1000,\n"
            "    \"convergence\": \"RELATIVE_INI_CORE\",\n"
            "    \"tolerance\": 1e-6\n"
            "  }\n"
            "}\n";
        std::cout << "Using AMG preconditioner\n\n";
    } else {
        config_str = 
            "{\n"
            "  \"config_version\": 2,\n"
            "  \"solver\": {\n"
            "    \"solver\": \"PCG\",\n"
            "    \"preconditioner\": \"BLOCK_JACOBI\",\n"
            "    \"print_solve_stats\": 1,\n"
            "    \"monitor_residual\": 1,\n"
            "    \"max_iters\": 1000,\n"
            "    \"convergence\": \"RELATIVE_INI_CORE\",\n"
            "    \"tolerance\": 1e-6\n"
            "  }\n"
            "}\n";
        std::cout << "Using JACOBI preconditioner\n\n";
    }
    
    const char* config_string = config_str.c_str();

    AMGX_config_handle config;
    check_amgx_error(AMGX_config_create(&config, config_string), 
                     "Config creation");

    AMGX_resources_handle resources;
    check_amgx_error(AMGX_resources_create_simple(&resources, config), 
                     "Resources creation");

    // 测试不同网格尺寸
    for (int grid_size : grid_sizes) {
        int nx = grid_size, ny = grid_size;
        int n = nx * ny;
        
        std::cout << "Testing grid size: " << nx << " x " << ny 
                  << " (" << n << " unknowns)" << std::endl;

        timer.reset();
        timer.start("total");
        timer.start("setup");

        // ⭐ 关键：在 GPU 上生成矩阵
        int *d_row_ptrs, *d_col_indices;
        double *d_values, *d_rhs;
        int nnz;
        
        generate_poisson_matrix_gpu(nx, ny, &d_row_ptrs, &d_col_indices, 
                                   &d_values, &d_rhs, &nnz);
        
        std::cout << "Matrix generated on GPU: " << n << " x " << n 
                  << ", nnz = " << nnz << std::endl;

        // 创建 AMGX 对象
        AMGX_matrix_handle A;
        AMGX_vector_handle x, b;
        AMGX_solver_handle solver;

        check_amgx_error(AMGX_matrix_create(&A, resources, AMGX_mode_dDDI), 
                         "Matrix creation");
        check_amgx_error(AMGX_vector_create(&x, resources, AMGX_mode_dDDI), 
                         "Vector x creation");
        check_amgx_error(AMGX_vector_create(&b, resources, AMGX_mode_dDDI), 
                         "Vector b creation");

        // ⭐ 关键：直接从 GPU 指针上传
        check_amgx_error(
            AMGX_matrix_upload_all(A, n, nnz, 1, 1,
                                  d_row_ptrs, d_col_indices, d_values, nullptr),
            "Matrix upload from device"
        );

        // 初始化解向量（在 GPU 上）
        double *d_x_init;
        cudaMalloc(&d_x_init, n * sizeof(double));
        cudaMemset(d_x_init, 0, n * sizeof(double));
        
        check_amgx_error(AMGX_vector_upload(x, n, 1, d_x_init), 
                         "Vector x upload");
        check_amgx_error(AMGX_vector_upload(b, n, 1, d_rhs), 
                         "Vector b upload from device");

        // 创建求解器
        check_amgx_error(AMGX_solver_create(&solver, resources, AMGX_mode_dDDI, config),
                         "Solver creation");

        // Setup
        check_amgx_error(AMGX_solver_setup(solver, A), "Solver setup");
        timer.stop("setup");

        // Solve
        timer.start("solve");
        check_amgx_error(AMGX_solver_solve(solver, b, x), "Solver solve");
        cudaDeviceSynchronize();
        timer.stop("solve");
        timer.stop("total");

        // 获取求解信息
        AMGX_SOLVE_STATUS status;
        check_amgx_error(AMGX_solver_get_status(solver, &status), 
                         "Get solver status");

        int iterations;
        check_amgx_error(AMGX_solver_get_iterations_number(solver, &iterations), 
                         "Get iterations");

        // 下载结果到 GPU（保持在 GPU 上）
        double *d_solution;
        cudaMalloc(&d_solution, n * sizeof(double));
        check_amgx_error(AMGX_vector_download(x, d_solution), 
                         "Download solution");

        // 计算残差（在 GPU 上）
        // 为简单起见，我们下载到 CPU 计算
        std::vector<double> x_result(n);
        cudaMemcpy(x_result.data(), d_solution, n * sizeof(double), 
                   cudaMemcpyDeviceToHost);
        
        std::vector<int> rows_cpu(n+1), cols_cpu(nnz);
        std::vector<double> vals_cpu(nnz), rhs_cpu(n);
        
        cudaMemcpy(rows_cpu.data(), d_row_ptrs, (n+1) * sizeof(int), 
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cols_cpu.data(), d_col_indices, nnz * sizeof(int), 
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vals_cpu.data(), d_values, nnz * sizeof(double), 
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(rhs_cpu.data(), d_rhs, n * sizeof(double), 
                   cudaMemcpyDeviceToHost);
        
        // 计算残差 r = b - Ax
        std::vector<double> residual(n, 0.0);
        for (int i = 0; i < n; i++) {
            double ax = 0.0;
            for (int j = rows_cpu[i]; j < rows_cpu[i+1]; j++) {
                ax += vals_cpu[j] * x_result[cols_cpu[j]];
            }
            residual[i] = rhs_cpu[i] - ax;
        }
        
        double residual_norm = 0.0;
        double rhs_norm = 0.0;
        for (int i = 0; i < n; i++) {
            residual_norm += residual[i] * residual[i];
            rhs_norm += rhs_cpu[i] * rhs_cpu[i];
        }
        residual_norm = sqrt(residual_norm);
        rhs_norm = sqrt(rhs_norm);
        double relative_residual = residual_norm / rhs_norm;

        // 保存结果
        ResultWriter::BenchmarkResult result;
        result.library_name = "AMGX-GPU-Native";
        result.test_name = "Poisson_2D_PCG_" + precond_type + "_GPUNative";
        result.problem_size = n;
        result.setup_time = timer.get_total_time("setup");
        result.solve_time = timer.get_total_time("solve");
        result.total_time = timer.get_total_time("total");
        result.iterations = iterations;
        result.residual = relative_residual;
        
        writer.add_result(result);

        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Relative residual: " << relative_residual << std::endl;
        std::cout << "Status: " << (status == AMGX_SOLVE_SUCCESS ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "Setup time: " << result.setup_time << " s" << std::endl;
        std::cout << "Solve time: " << result.solve_time << " s" << std::endl;
        std::cout << "Total time: " << result.total_time << " s" << std::endl;
        std::cout << std::endl;

        // 清理
        AMGX_solver_destroy(solver);
        AMGX_matrix_destroy(A);
        AMGX_vector_destroy(b);
        AMGX_vector_destroy(x);
        
        cudaFree(d_row_ptrs);
        cudaFree(d_col_indices);
        cudaFree(d_values);
        cudaFree(d_rhs);
        cudaFree(d_x_init);
        cudaFree(d_solution);
    }

    // 保存结果到文件
    std::cout << "Saving results..." << std::endl;
    writer.write_csv("amgx_gpu_native_results.csv");
    writer.write_json("amgx_gpu_native_results.json");
    std::cout << "Results saved to ../results/" << std::endl;

    // 清理全局资源
    AMGX_resources_destroy(resources);
    AMGX_config_destroy(config);
    AMGX_finalize_plugins();
    AMGX_finalize();

    return 0;
}

