// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MOUNTAIN_LAKE_LAYERS_SIGMOID_H_
#define MOUNTAIN_LAKE_LAYERS_SIGMOID_H_

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXf;

/// @brief sigmoid函数类（Class of sigmoid functions）
class Sigmoid {
 public:
  explicit Sigmoid(){};
  ~Sigmoid(){};
  void Forward(MatrixXf &A, MatrixXf &Z);
  void Backward(MatrixXf &dZ, MatrixXf &Y, MatrixXf &dA);
};

#endif  // MOUNTAIN_LAKE_LAYERS_SIGMOID_H_