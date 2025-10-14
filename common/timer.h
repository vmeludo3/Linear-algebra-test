#ifndef BENCHMARK_TIMER_H
#define BENCHMARK_TIMER_H

#include <chrono>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

/**
 * 高精度计时器类，用于性能测试
 */
class BenchmarkTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double>;

private:
    std::map<std::string, TimePoint> start_times_;
    std::map<std::string, double> elapsed_times_;
    std::map<std::string, int> call_counts_;

public:
    /**
     * 开始计时
     */
    void start(const std::string& name) {
        start_times_[name] = Clock::now();
    }

    /**
     * 停止计时并记录
     */
    void stop(const std::string& name) {
        auto end_time = Clock::now();
        if (start_times_.find(name) != start_times_.end()) {
            Duration elapsed = end_time - start_times_[name];
            elapsed_times_[name] += elapsed.count();
            call_counts_[name]++;
        }
    }

    /**
     * 获取总时间
     */
    double get_total_time(const std::string& name) const {
        auto it = elapsed_times_.find(name);
        return (it != elapsed_times_.end()) ? it->second : 0.0;
    }

    /**
     * 获取平均时间
     */
    double get_average_time(const std::string& name) const {
        auto it = elapsed_times_.find(name);
        auto count_it = call_counts_.find(name);
        if (it != elapsed_times_.end() && count_it != call_counts_.end() && count_it->second > 0) {
            return it->second / count_it->second;
        }
        return 0.0;
    }

    /**
     * 获取调用次数
     */
    int get_call_count(const std::string& name) const {
        auto it = call_counts_.find(name);
        return (it != call_counts_.end()) ? it->second : 0;
    }

    /**
     * 重置所有计时器
     */
    void reset() {
        start_times_.clear();
        elapsed_times_.clear();
        call_counts_.clear();
    }

    /**
     * 重置特定计时器
     */
    void reset(const std::string& name) {
        start_times_.erase(name);
        elapsed_times_.erase(name);
        call_counts_.erase(name);
    }

    /**
     * 打印所有计时结果
     */
    void print_summary() const {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "Performance Summary\n";
        std::cout << std::string(70, '=') << "\n";
        std::cout << std::left << std::setw(30) << "Timer Name" 
                  << std::right << std::setw(12) << "Total (s)" 
                  << std::setw(12) << "Avg (s)" 
                  << std::setw(10) << "Calls" << "\n";
        std::cout << std::string(70, '-') << "\n";

        for (const auto& pair : elapsed_times_) {
            const std::string& name = pair.first;
            double total = pair.second;
            int count = call_counts_.at(name);
            double avg = total / count;

            std::cout << std::left << std::setw(30) << name 
                      << std::right << std::setw(12) << std::fixed << std::setprecision(6) << total
                      << std::setw(12) << std::fixed << std::setprecision(6) << avg
                      << std::setw(10) << count << "\n";
        }
        std::cout << std::string(70, '=') << "\n\n";
    }
};

/**
 * RAII 风格的计时器，自动开始和停止
 */
class ScopedTimer {
private:
    BenchmarkTimer& timer_;
    std::string name_;

public:
    ScopedTimer(BenchmarkTimer& timer, const std::string& name) 
        : timer_(timer), name_(name) {
        timer_.start(name_);
    }

    ~ScopedTimer() {
        timer_.stop(name_);
    }
};

#endif // BENCHMARK_TIMER_H

