// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MOUNTAIN_LAKE_LAYERS_RELU_H_
#define MOUNTAIN_LAKE_LAYERS_RELU_H_

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXf;

/// @brief 线性整流函数类（Class of linear rectifier functions）
class ReLU {
 public:
  ReLU(){};
  ~ReLU(){};
  void Forward(MatrixXf &A, MatrixXf &Z);
  void Backward(MatrixXf &dZ, MatrixXf &Y, MatrixXf &dA);
};

#endif  // MOUNTAIN_LAKE_LAYERS_RELU_H_