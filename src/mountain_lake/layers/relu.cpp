// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "relu.h"

/// @brief 线性整流层正向传播（Linear rectifier forward propagation）
/// @param A 输入（input）
/// @param Z 输出（output）
void ReLU::Forward(MatrixXf &A, MatrixXf &Z) {
  Z = (A.array() > 0).select(A, 0);
}

/// @brief 线性整流层反向传播（Linear rectifier backpropagation）
/// @param dZ 输出信号的导数（Derivative of the output signal）
/// @param Z 输出（output）
/// @param dA 输入信号的导数（Derivative of the input signal）
void ReLU::Backward(MatrixXf &dZ, MatrixXf &Z, MatrixXf &dA) {
  dA = (Z.array() > 0).select(dZ, 0);
}