cmake_minimum_required(VERSION 3.20)

# project在哪里，${PROJECT_SOURCE_DIR}就在哪里
project(mountain_lake_test LANGUAGES CXX)

# 目前的 GoogleTest 至少需要 C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 启用GCC所有警告，视警告为错误
# -O3 表示启用最高优化级别
# -fopenmp 表示启用OpenMP支持，如果不需要并行，可以去掉
# -mavx -mfma 表示启用AVX（Advanced Vector Extensions）和FMA（Fused Multiply-Add）指令集支持
# -mavx：启用AVX指令集，使程序可以利用CPU的AVX硬件支持。AVX指令集通常用于处理浮点数和整数向量，以及进行逻辑运算和算术运算。
#       启用AVX可以提高程序的性能，特别是当程序中包含大量向量运算时。
# -mfma：启用FMA指令集，使程序可以利用CPU的FMA硬件支持。FMA指令集可以同时执行加法和乘法操作，从而提高程序的性能。
#       FMA指令集通常用于处理浮点数和整数向量，以及进行逻辑运算和算术运算。启用FMA可以提高程序中向量运算的性能。
#       需要注意的是，这些编译选项需要在特定的编译器环境中使用，并且可能需要同时启用-mavx和-mfma选项。
# -Werror 表示将所有警告视为错误
# -Wall -Wextra 表示启用额外的警告
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -O3 -fopenmp -mavx -mfma")

# 使用enable_testing()后，在build目录中会生成一个Testing文件夹，
# 里面包含测试函数的实现。然后在build目录下执行ctest命令才生效。
# 注意：在执行ctest命令时，需要进入build目录，而不是项目根目录。
# 这里的测试内容与 add_test 设置有关。
enable_testing()

add_subdirectory(tests)