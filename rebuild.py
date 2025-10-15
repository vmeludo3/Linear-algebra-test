#!/usr/bin/env python3
"""
LinearAlgebra_cuda 构建脚本
用途: 清空旧构建、配置 CMake、重新编译
作者: Auto-generated
日期: 2025-10-14
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# 颜色输出
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(msg):
    """打印标题"""
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*70}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{msg:^70}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*70}{Color.END}\n")

def print_step(step, msg):
    """打印步骤"""
    print(f"{Color.BOLD}{Color.BLUE}[步骤 {step}]{Color.END} {msg}")

def print_success(msg):
    """打印成功消息"""
    print(f"{Color.GREEN}✅ {msg}{Color.END}")

def print_error(msg):
    """打印错误消息"""
    print(f"{Color.RED}❌ {msg}{Color.END}")

def print_warning(msg):
    """打印警告消息"""
    print(f"{Color.YELLOW}⚠️  {msg}{Color.END}")

def get_project_root():
    """获取项目根目录"""
    script_dir = Path(__file__).parent.absolute()
    return script_dir

def read_env_vars():
    """读取环境变量中的库路径"""
    env_vars = {}
    required_vars = ['AMGX_DIR', 'HYPRE_CPU_DIR', 'HYPRE_GPU_DIR', 'PETSC_DIR', 'YAML_CPP_DIR']
    
    print_step(0, "检查环境变量")
    all_set = True
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            env_vars[var] = value
            print(f"  ✅ {var:15} = {value}")
        else:
            print_warning(f"{var} 未设置")
            all_set = False
    
    if all_set:
        print_success("所有环境变量已设置")
        return env_vars
    else:
        print_warning("部分环境变量未设置，将使用 CMakeLists.txt 中的默认值")
        print("  提示: 运行 'source env_setup.sh' 来设置环境变量")
        return None

def clean_build(build_dir):
    """清理构建目录"""
    print_step(1, "清理旧构建文件")
    
    if build_dir.exists():
        print(f"  删除目录: {build_dir}")
        shutil.rmtree(build_dir)
        print_success("旧构建文件已清理")
    else:
        print("  构建目录不存在，无需清理")
    
    # 创建新的构建目录
    build_dir.mkdir(parents=True, exist_ok=True)
    print(f"  创建目录: {build_dir}")

def run_cmake(build_dir, env_vars=None, build_type="Release"):
    """运行 CMake 配置"""
    print_step(2, f"配置 CMake (构建类型: {build_type})")
    
    # 构建 CMake 命令
    cmake_cmd = ["cmake", ".."]
    
    # 设置构建类型
    cmake_cmd.extend(["-DCMAKE_BUILD_TYPE=" + build_type])
    
    # 如果提供了环境变量，添加到命令中
    if env_vars:
        print("  使用环境变量配置库路径:")
        for var, value in env_vars.items():
            cmake_cmd.append(f"-D{var}={value}")
            print(f"    {var} = {value}")
    else:
        print("  使用 CMakeLists.txt 中的默认路径")
    
    # 运行 CMake
    print(f"\n  运行: {' '.join(cmake_cmd)}\n")
    
    try:
        result = subprocess.run(
            cmake_cmd,
            cwd=build_dir,
            check=True,
            capture_output=False
        )
        print_success("CMake 配置成功")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"CMake 配置失败: {e}")
        return False

def run_make(build_dir, jobs=None):
    """运行 Make 编译"""
    print_step(3, "编译项目")
    
    # 确定并行任务数
    if jobs is None:
        try:
            jobs = os.cpu_count() or 4
        except:
            jobs = 4
    
    print(f"  使用 {jobs} 个并行任务")
    
    # 运行 Make
    make_cmd = ["make", f"-j{jobs}"]
    print(f"  运行: {' '.join(make_cmd)}\n")
    
    try:
        result = subprocess.run(
            make_cmd,
            cwd=build_dir,
            check=True,
            capture_output=False
        )
        print_success("编译成功")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"编译失败: {e}")
        return False

def list_executables(build_dir):
    """列出生成的可执行文件"""
    print_step(4, "列出生成的可执行文件")
    
    executables = []
    
    # 查找所有可执行文件
    test_dirs = [
        "amgx_tests",
        "amgx_gpu_native_tests",
        "hypre_tests",
        "hypre_gpu_tests",
        "hypre_gpu_native_tests",
        "petsc_tests",
        "petsc_gpu_tests",
        "petsc_gpu_native_tests"
    ]
    
    for test_dir in test_dirs:
        dir_path = build_dir / test_dir
        if dir_path.exists():
            for exe in dir_path.glob("*"):
                if exe.is_file() and os.access(exe, os.X_OK):
                    executables.append(exe)
    
    if executables:
        print(f"\n  找到 {len(executables)} 个可执行文件:\n")
        for exe in sorted(executables):
            rel_path = exe.relative_to(build_dir)
            size = exe.stat().st_size / (1024 * 1024)  # MB
            print(f"    ✅ ./{rel_path} ({size:.1f} MB)")
        print()
    else:
        print_warning("未找到可执行文件")
    
    return executables

def show_usage():
    """显示使用说明"""
    print(f"\n{Color.BOLD}{'─'*70}{Color.END}")
    print(f"{Color.BOLD}下一步操作:{Color.END}\n")
    print("  1. 运行测试:")
    print(f"     {Color.CYAN}cd build{Color.END}")
    print(f"     {Color.CYAN}./amgx_gpu_native_tests/amgx_gpu_native_poisson_solver{Color.END}")
    print()
    print("  2. 查看结果:")
    print(f"     {Color.CYAN}cd results{Color.END}")
    print(f"     {Color.CYAN}cat amgx_gpu_native_results.csv{Color.END}")
    print()
    print("  3. 修改配置:")
    print(f"     {Color.CYAN}vim solver_config.yaml{Color.END}")
    print(f"     {Color.CYAN}python3 rebuild.py{Color.END}")
    print(f"{Color.BOLD}{'─'*70}{Color.END}\n")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="LinearAlgebra_cuda 构建脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 rebuild.py                    # 标准构建
  python3 rebuild.py --clean-only       # 仅清理构建
  python3 rebuild.py --jobs 8           # 使用 8 个并行任务
  python3 rebuild.py --debug            # Debug 模式编译
  python3 rebuild.py --use-env          # 使用环境变量配置路径
        """
    )
    
    parser.add_argument('--clean-only', action='store_true',
                        help='仅清理构建目录，不重新编译')
    parser.add_argument('--jobs', '-j', type=int, default=None,
                        help='并行编译任务数（默认：CPU 核心数）')
    parser.add_argument('--build-type', choices=['Release', 'Debug'], default='Release',
                        help='构建类型（默认：Release）')
    parser.add_argument('--use-env', action='store_true',
                        help='使用环境变量中的库路径（需先 source env_setup.sh）')
    parser.add_argument('--no-cmake', action='store_true',
                        help='跳过 CMake 配置，仅编译')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = get_project_root()
    build_dir = project_root / "build"
    
    print_header("LinearAlgebra_cuda 构建脚本")
    print(f"项目目录: {project_root}")
    print(f"构建目录: {build_dir}\n")
    
    # 读取环境变量（如果需要）
    env_vars = None
    if args.use_env:
        env_vars = read_env_vars()
        if env_vars is None:
            print_error("使用 --use-env 但环境变量未完全设置")
            print("  请先运行: source env_setup.sh")
            sys.exit(1)
    
    # 步骤 1: 清理
    if not args.no_cmake:
        clean_build(build_dir)
    
    # 如果仅清理，退出
    if args.clean_only:
        print_success("清理完成！")
        sys.exit(0)
    
    # 步骤 2: CMake 配置
    if not args.no_cmake:
        if not run_cmake(build_dir, env_vars, args.build_type):
            print_error("构建失败！")
            sys.exit(1)
    else:
        print_warning("跳过 CMake 配置（--no-cmake）")
    
    # 步骤 3: 编译
    if not run_make(build_dir, args.jobs):
        print_error("构建失败！")
        sys.exit(1)
    
    # 步骤 4: 列出可执行文件
    executables = list_executables(build_dir)
    
    # 显示使用说明
    if executables:
        show_usage()
        print_success("构建完成！")
    else:
        print_error("构建完成但未生成可执行文件")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}⚠️  构建被用户中断{Color.END}")
        sys.exit(130)
    except Exception as e:
        print_error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

