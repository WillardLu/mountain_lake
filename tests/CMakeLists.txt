find_package(GTest REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(mountain_lake_test
  mountain_lake_test.cpp)

target_link_libraries(mountain_lake_test
 PRIVATE
  GTest::GTest)

target_include_directories(mountain_lake_test
 PUBLIC
  ${PROJECT_SOURCE_DIR}/src/)
