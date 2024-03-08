// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MOUNTAIN_LAKE_LAYERS_CONVOLUTION_H_
#define MOUNTAIN_LAKE_LAYERS_CONVOLUTION_H_

#include <eigen3/Eigen/Dense>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::RowMajor;

/// @brief 卷积层配置结构（Convolutional Layer Configuration Structure）
struct ConvConig {
  int pad = 0;
  int stride = 0;
  int height = 0;
  int width = 0;
  int number = 0;
  int channel_num = 0;
  int i_height = 0;
  int i_width = 0;
  int o_height = 0;
  int o_width = 0;
};

// 卷积层类（Convolutional Layer Class）
class Convolution {
 public:
  Convolution(){};
  ~Convolution(){};
  void Forward(MatrixXf &X, MatrixXf &W, MatrixXf &B, MatrixXf &O,
               ConvConig &cc);
  void Backward(MatrixXf &X, MatrixXf &dB, MatrixXf &dO, MatrixXf &dW,
                ConvConig &cc, int layer_num);

 private:
};

#endif  // MOUNTAIN_LAKE_LAYERS_CONVOLUTION_H_