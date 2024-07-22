// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "sigmoid.h"

/// @brief sigmoid层正向传播（Forward propagation of the sigmoid layer）
/// @param A 输入（input）
/// @param Z 输出（output）
void Sigmoid::Forward(MatrixXf &A, MatrixXf &Z) {
  Z = 1 / (1 + (-A).array().exp());
}

/// @brief sigamoid层反向传播（sigamoid layer backpropagation）
/// @param dZ 输出信号的导数（Derivative of the output signal）
/// @param Z 输出（output）
/// @param dA 输入信号的导数（Derivative of the input signal）
void Sigmoid::Backward(MatrixXf &dZ, MatrixXf &Z, MatrixXf &dA) {
  dA = dZ.array() * Z.array() * (1 - Z.array());
}