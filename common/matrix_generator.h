#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <vector>
#include <cmath>

/**
 * 矩阵生成器 - 用于生成各种测试矩阵
 */
class MatrixGenerator {
public:
    /**
     * 生成 2D Poisson 问题的 5 点模板
     * @param nx x 方向网格点数
     * @param ny y 方向网格点数
     * @param rows 输出行索引
     * @param cols 输出列索引
     * @param values 输出矩阵值
     * @param rhs 输出右端项
     */
    static void generate_2d_poisson_5pt(int nx, int ny,
                                         std::vector<int>& rows,
                                         std::vector<int>& cols,
                                         std::vector<double>& values,
                                         std::vector<double>& rhs) {
        int n = nx * ny;
        rhs.resize(n, 1.0);  // 右端项设为 1
        
        rows.clear();
        cols.clear();
        values.clear();
        
        // 预分配空间
        rows.reserve(n * 5);
        cols.reserve(n * 5);
        values.reserve(n * 5);
        
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                
                // 对角元素
                rows.push_back(idx);
                cols.push_back(idx);
                values.push_back(4.0);
                
                // 左边
                if (i > 0) {
                    rows.push_back(idx);
                    cols.push_back(idx - 1);
                    values.push_back(-1.0);
                }
                
                // 右边
                if (i < nx - 1) {
                    rows.push_back(idx);
                    cols.push_back(idx + 1);
                    values.push_back(-1.0);
                }
                
                // 下方
                if (j > 0) {
                    rows.push_back(idx);
                    cols.push_back(idx - nx);
                    values.push_back(-1.0);
                }
                
                // 上方
                if (j < ny - 1) {
                    rows.push_back(idx);
                    cols.push_back(idx + nx);
                    values.push_back(-1.0);
                }
            }
        }
    }
    
    /**
     * 生成 3D Poisson 问题的 7 点模板
     * @param nx x 方向网格点数
     * @param ny y 方向网格点数
     * @param nz z 方向网格点数
     */
    static void generate_3d_poisson_7pt(int nx, int ny, int nz,
                                         std::vector<int>& rows,
                                         std::vector<int>& cols,
                                         std::vector<double>& values,
                                         std::vector<double>& rhs) {
        int n = nx * ny * nz;
        rhs.resize(n, 1.0);
        
        rows.clear();
        cols.clear();
        values.clear();
        
        rows.reserve(n * 7);
        cols.reserve(n * 7);
        values.reserve(n * 7);
        
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int idx = k * nx * ny + j * nx + i;
                    
                    // 对角元素
                    rows.push_back(idx);
                    cols.push_back(idx);
                    values.push_back(6.0);
                    
                    // x 方向
                    if (i > 0) {
                        rows.push_back(idx);
                        cols.push_back(idx - 1);
                        values.push_back(-1.0);
                    }
                    if (i < nx - 1) {
                        rows.push_back(idx);
                        cols.push_back(idx + 1);
                        values.push_back(-1.0);
                    }
                    
                    // y 方向
                    if (j > 0) {
                        rows.push_back(idx);
                        cols.push_back(idx - nx);
                        values.push_back(-1.0);
                    }
                    if (j < ny - 1) {
                        rows.push_back(idx);
                        cols.push_back(idx + nx);
                        values.push_back(-1.0);
                    }
                    
                    // z 方向
                    if (k > 0) {
                        rows.push_back(idx);
                        cols.push_back(idx - nx * ny);
                        values.push_back(-1.0);
                    }
                    if (k < nz - 1) {
                        rows.push_back(idx);
                        cols.push_back(idx + nx * ny);
                        values.push_back(-1.0);
                    }
                }
            }
        }
    }
    
    /**
     * 计算解的 L2 范数误差
     */
    static double compute_l2_error(const std::vector<double>& computed,
                                    const std::vector<double>& exact) {
        if (computed.size() != exact.size()) {
            return -1.0;
        }
        
        double error = 0.0;
        double norm = 0.0;
        
        for (size_t i = 0; i < computed.size(); i++) {
            double diff = computed[i] - exact[i];
            error += diff * diff;
            norm += exact[i] * exact[i];
        }
        
        return std::sqrt(error) / std::sqrt(norm);
    }
};

#endif // MATRIX_GENERATOR_H

