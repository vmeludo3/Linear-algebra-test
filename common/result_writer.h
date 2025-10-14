#ifndef RESULT_WRITER_H
#define RESULT_WRITER_H

#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <iomanip>
#include <ctime>

/**
 * 结果写入器 - 将测试结果保存为 CSV 和 JSON 格式
 */
class ResultWriter {
public:
    struct BenchmarkResult {
        std::string library_name;
        std::string test_name;
        int problem_size;
        double setup_time;
        double solve_time;
        double total_time;
        int iterations;
        double residual;
        std::map<std::string, std::string> extra_info;
    };

private:
    std::string output_dir_;
    std::vector<BenchmarkResult> results_;

public:
    ResultWriter(const std::string& output_dir = "../results") 
        : output_dir_(output_dir) {}

    /**
     * 添加一个测试结果
     */
    void add_result(const BenchmarkResult& result) {
        results_.push_back(result);
    }

    /**
     * 写入 CSV 文件
     */
    void write_csv(const std::string& filename) const {
        std::ofstream file(output_dir_ + "/" + filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // 写入表头
        file << "Library,Test,ProblemSize,SetupTime(s),SolveTime(s),TotalTime(s),"
             << "Iterations,Residual\n";

        // 写入数据
        for (const auto& result : results_) {
            file << result.library_name << ","
                 << result.test_name << ","
                 << result.problem_size << ","
                 << std::fixed << std::setprecision(6) << result.setup_time << ","
                 << std::fixed << std::setprecision(6) << result.solve_time << ","
                 << std::fixed << std::setprecision(6) << result.total_time << ","
                 << result.iterations << ","
                 << std::scientific << std::setprecision(6) << result.residual << "\n";
        }

        file.close();
        std::cout << "Results written to: " << output_dir_ + "/" + filename << std::endl;
    }

    /**
     * 写入 JSON 文件
     */
    void write_json(const std::string& filename) const {
        std::ofstream file(output_dir_ + "/" + filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        file << "{\n";
        file << "  \"timestamp\": \"" << get_timestamp() << "\",\n";
        file << "  \"results\": [\n";

        for (size_t i = 0; i < results_.size(); i++) {
            const auto& result = results_[i];
            file << "    {\n";
            file << "      \"library\": \"" << result.library_name << "\",\n";
            file << "      \"test\": \"" << result.test_name << "\",\n";
            file << "      \"problem_size\": " << result.problem_size << ",\n";
            file << "      \"setup_time\": " << std::fixed << std::setprecision(6) 
                 << result.setup_time << ",\n";
            file << "      \"solve_time\": " << std::fixed << std::setprecision(6) 
                 << result.solve_time << ",\n";
            file << "      \"total_time\": " << std::fixed << std::setprecision(6) 
                 << result.total_time << ",\n";
            file << "      \"iterations\": " << result.iterations << ",\n";
            file << "      \"residual\": " << std::scientific << std::setprecision(6) 
                 << result.residual << "\n";
            file << "    }" << (i < results_.size() - 1 ? "," : "") << "\n";
        }

        file << "  ]\n";
        file << "}\n";

        file.close();
        std::cout << "Results written to: " << output_dir_ + "/" + filename << std::endl;
    }

    /**
     * 打印结果摘要到控制台
     */
    void print_summary() const {
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "Benchmark Results Summary\n";
        std::cout << std::string(100, '=') << "\n";
        std::cout << std::left << std::setw(15) << "Library" 
                  << std::setw(25) << "Test"
                  << std::right << std::setw(12) << "Size"
                  << std::setw(12) << "Setup(s)"
                  << std::setw(12) << "Solve(s)"
                  << std::setw(10) << "Iters"
                  << std::setw(12) << "Residual" << "\n";
        std::cout << std::string(100, '-') << "\n";

        for (const auto& result : results_) {
            std::cout << std::left << std::setw(15) << result.library_name
                      << std::setw(25) << result.test_name
                      << std::right << std::setw(12) << result.problem_size
                      << std::setw(12) << std::fixed << std::setprecision(4) << result.setup_time
                      << std::setw(12) << std::fixed << std::setprecision(4) << result.solve_time
                      << std::setw(10) << result.iterations
                      << std::setw(12) << std::scientific << std::setprecision(2) << result.residual << "\n";
        }
        std::cout << std::string(100, '=') << "\n\n";
    }

private:
    /**
     * 获取当前时间戳
     */
    std::string get_timestamp() const {
        std::time_t now = std::time(nullptr);
        char buf[100];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buf);
    }
};

#endif // RESULT_WRITER_H

