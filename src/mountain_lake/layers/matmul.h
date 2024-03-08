// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MOUNTAIN_LAKE_LAYERS_MATMUL_H_
#define MOUNTAIN_LAKE_LAYERS_MATMUL_H_

#include <mountain_town/math/random.h>

#include <eigen3/Eigen/Dense>
#include <string>

using Eigen::MatrixXf;
using std::string;

/// @brief 仿射变换层类（class of affine transformation layers）
class MatMul {
 public:
  MatMul(){};
  ~MatMul(){};
  void Forward(MatrixXf &X, MatrixXf &W, MatrixXf &A);
  void Backward(MatrixXf &X, MatrixXf &W, MatrixXf &dA, MatrixXf &dW,
                MatrixXf &dX, int layer_num);
};

#endif  // MOUNTAIN_LAKE_LAYERS_MATMUL_H_