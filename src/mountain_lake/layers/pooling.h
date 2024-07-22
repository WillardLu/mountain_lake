// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MOUNTAIN_LAKE_LAYERS_POOLING_H_
#define MOUNTAIN_LAKE_LAYERS_POOLING_H_

#include <eigen3/Eigen/Dense>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::RowMajor;

/// @brief 池化层配置结构（Pooling layer configuration structure）
struct PoolConfig {
  int height = 0;
  int width = 0;
  int stride = 0;
  // 0为取最大值，1为取平均值
  // 0 is the maximum value, 1 is the average value.
  int type = 0;
  int filter_num = 0;
  int i_height = 0;
  int i_width = 0;
  int o_height = 0;
  int o_width = 0;
};

// 池化层类（pooling layer class）
class Pooling {
 public:
  Pooling(){};
  ~Pooling(){};
  void Forward(MatrixXf &A, MatrixXf &O, PoolConfig &pc);
  void Backward(MatrixXf &dZ, MatrixXf &dA, MatrixXf &A, PoolConfig &pc);

 private:
};

#endif  // MOUNTAIN_LAKE_LAYERS_POOLING_H_