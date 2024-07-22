// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "affine.h"

/// @brief 仿射变换层正向传播
///        （Forward propagation of affine transformed layers）
/// @param X 输入信号（input signals）
/// @param W 权重（weights）
/// @param B 偏置（bias）
/// @param A 输出信号（output signals）
void Affine::Forward(MatrixXf &X, MatrixXf &W, MatrixXf &B, MatrixXf &A) {
  A = X * W + B;
}

/// @brief 仿射变换层反向传播
///        （Backpropagation of affine transformed layers）
/// @param X 输入信号（input signals）
/// @param W 权重（weights）
/// @param dA 输出信号的导数（derivative of the output signal）
/// @param dB 偏置的导数（derivative of bias）
/// @param dW 权重的导数（derivative of weights)
/// @param dX 输入信号的导数（derivative of the input signal）
/// @param layer_num 所处层号（Layer number）
void Affine::Backward(MatrixXf &X, MatrixXf &W, MatrixXf &dA, MatrixXf &dB,
                      MatrixXf &dW, MatrixXf &dX, int layer_num) {
  dB.noalias() = dA;
  dW.noalias() = X.transpose() * dA;
  // 如果这个层被放在神经网络中的第一层，则不需要计算输入信号的导数。
  // If this layer is placed in the first layer in the neural network, there is
  // no need to calculate the derivative of the input signal.
  if (layer_num >= 2) dX.noalias() = dA * W.transpose();
}