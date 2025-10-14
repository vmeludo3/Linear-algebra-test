#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <string>
#include <vector>
#include <iostream>
#include <yaml-cpp/yaml.h>

/**
 * 配置文件读取器
 * 使用 YAML 格式配置求解器参数
 */
class ConfigReader {
private:
    YAML::Node config_;
    std::string config_file_;

public:
    /**
     * 构造函数：加载配置文件
     */
    ConfigReader(const std::string& config_file = "../solver_config.yaml") 
        : config_file_(config_file) {
        try {
            config_ = YAML::LoadFile(config_file_);
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading config file: " << config_file_ << std::endl;
            std::cerr << "Error: " << e.what() << std::endl;
            std::cerr << "Using default configuration." << std::endl;
        }
    }

    /**
     * 获取 AMGX 预条件器类型
     */
    std::string get_amgx_preconditioner() const {
        try {
            return config_["amgx"]["preconditioner"].as<std::string>();
        } catch (...) {
            return "AMG";  // 默认值
        }
    }

    /**
     * 获取 AMGX 求解器类型
     */
    std::string get_amgx_solver() const {
        try {
            return config_["amgx"]["solver"].as<std::string>();
        } catch (...) {
            return "PCG";
        }
    }

    /**
     * 获取 HYPRE 预条件器类型
     */
    std::string get_hypre_preconditioner(bool is_gpu = false) const {
        try {
            std::string key = is_gpu ? "hypre_gpu" : "hypre_cpu";
            return config_[key]["preconditioner"].as<std::string>();
        } catch (...) {
            return "BOOMERAMG";
        }
    }

    /**
     * 获取 PETSc 预条件器类型
     */
    std::string get_petsc_preconditioner(bool is_gpu = false) const {
        try {
            std::string key = is_gpu ? "petsc_gpu" : "petsc_cpu";
            return config_[key]["preconditioner"].as<std::string>();
        } catch (...) {
            return "GAMG";
        }
    }

    /**
     * 获取网格规模列表
     */
    std::vector<int> get_grid_sizes() const {
        std::vector<int> sizes;
        try {
            YAML::Node grid_sizes = config_["global"]["problem"]["grid_sizes"];
            for (size_t i = 0; i < grid_sizes.size(); i++) {
                sizes.push_back(grid_sizes[i].as<int>());
            }
        } catch (...) {
            // 默认值
            sizes = {64, 128, 256, 512};
        }
        return sizes;
    }

    /**
     * 获取最大迭代次数
     */
    int get_max_iters(const std::string& solver) const {
        try {
            return config_[solver]["convergence"]["max_iters"].as<int>();
        } catch (...) {
            return 1000;
        }
    }

    /**
     * 获取收敛容差
     */
    double get_tolerance(const std::string& solver) const {
        try {
            return config_[solver]["convergence"]["tolerance"].as<double>();
        } catch (...) {
            return 1e-6;
        }
    }

    /**
     * 打印配置摘要
     */
    void print_summary(const std::string& solver_name) const {
        std::cout << "Configuration for " << solver_name << ":\n";
        std::cout << "  Config file: " << config_file_ << "\n";
        
        try {
            if (solver_name == "AMGX") {
                std::cout << "  Solver: " << get_amgx_solver() << "\n";
                std::cout << "  Preconditioner: " << get_amgx_preconditioner() << "\n";
            } else if (solver_name.find("HYPRE") != std::string::npos) {
                bool is_gpu = (solver_name.find("GPU") != std::string::npos);
                std::cout << "  Solver: " << (is_gpu ? config_["hypre_gpu"]["solver"].as<std::string>() 
                                                     : config_["hypre_cpu"]["solver"].as<std::string>()) << "\n";
                std::cout << "  Preconditioner: " << get_hypre_preconditioner(is_gpu) << "\n";
            } else if (solver_name.find("PETSc") != std::string::npos) {
                bool is_gpu = (solver_name.find("GPU") != std::string::npos);
                std::string key = is_gpu ? "petsc_gpu" : "petsc_cpu";
                std::cout << "  Solver: " << config_[key]["solver"].as<std::string>() << "\n";
                std::cout << "  Preconditioner: " << get_petsc_preconditioner(is_gpu) << "\n";
            }
            
            auto sizes = get_grid_sizes();
            std::cout << "  Grid sizes: ";
            for (size_t i = 0; i < sizes.size(); i++) {
                std::cout << sizes[i];
                if (i < sizes.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
        } catch (const YAML::Exception& e) {
            std::cout << "  (Using defaults)\n";
        }
        std::cout << std::endl;
    }

    /**
     * 检查配置文件是否存在
     */
    bool is_loaded() const {
        return !config_.IsNull();
    }
};

#endif // CONFIG_READER_H

