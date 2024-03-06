find_package(GTest REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(mountain_lake_test
  mountain_lake_test.cpp
  neural_network/neural_network_test.cpp
  layers/affine_test.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/math/random.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/string/basic.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/string/toml.cpp
  ${PROJECT_SOURCE_DIR}/mountain_lake/layers/affine.cpp
  ${PROJECT_SOURCE_DIR}/mountain_lake/neural_network/neural_network.cpp)

target_link_libraries(mountain_lake_test
 PRIVATE
  GTest::GTest)

target_include_directories(mountain_lake_test
 PUBLIC
  ${PROJECT_SOURCE_DIR}/mountain_lake/
  ${PROJECT_SOURCE_DIR}/lib/)