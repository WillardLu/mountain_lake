find_package(GTest REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(SOURCES
  neural_network/neural_network_test.cpp
  layers/affine_test.cpp
  layers/convolution_test.cpp
  layers/matmul_test.cpp
  layers/pooling_test.cpp
  layers/relu_test.cpp
  layers/sigmoid_test.cpp
  layers/softmaxwithloss_test.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/math/random.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/string/basic.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/string/toml.cpp
  ${PROJECT_SOURCE_DIR}/src/mountain_lake/layers/affine.cpp
  ${PROJECT_SOURCE_DIR}/src/mountain_lake/layers/convolution.cpp
  ${PROJECT_SOURCE_DIR}/src/mountain_lake/layers/matmul.cpp
  ${PROJECT_SOURCE_DIR}/src/mountain_lake/layers/pooling.cpp
  ${PROJECT_SOURCE_DIR}/src/mountain_lake/layers/relu.cpp
  ${PROJECT_SOURCE_DIR}/src/mountain_lake/layers/sigmoid.cpp
  ${PROJECT_SOURCE_DIR}/src/mountain_lake/layers/softmaxwithloss.cpp
  ${PROJECT_SOURCE_DIR}/src/mountain_lake/neural_network/neural_network.cpp)

add_executable(mountain_lake_test mountain_lake_test.cpp)
target_sources(mountain_lake_test PRIVATE ${SOURCES})
set_property(TARGET mountain_lake_test PROPERTY CXX_STANDARD 17)

target_link_libraries(mountain_lake_test
 PRIVATE
  GTest::GTest)

target_include_directories(mountain_lake_test
 PRIVATE
  ${PROJECT_SOURCE_DIR}/src/
  ${PROJECT_SOURCE_DIR}/lib/)
