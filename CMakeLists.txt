cmake_minimum_required(VERSION 3.20)

project(mountain_lake LANGUAGES CXX)

# 目前的 GoogleTest 至少需要 C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 启用GCC所有警告，视警告为错误
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

# 使用enable_testing()后，在build目录中会生成一个Testing文件夹，
# 里面包含测试函数的实现。然后在build目录下执行ctest命令才生效。
# 注意：在执行ctest命令时，需要进入build目录，而不是项目根目录。
# 这里的测试内容与 add_test 设置有关。
enable_testing()
add_subdirectory(tests)
