// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "sigmoid.h"

/// @brief sigmoid层正向传播
/// @param A 输入信号
/// @param Z 输出信号
void Sigmoid::Forward(MatrixXf &A, MatrixXf &Z) {
  Z = 1 / (1 + (-A).array().exp());
}

/// @brief sigamoid层反向传播
/// @param dZ 输出信号的导数
/// @param Z 输出信号
/// @param dA 输入信号的导数
void Sigmoid::Backward(MatrixXf &dZ, MatrixXf &Z, MatrixXf &dA) {
  dA = dZ.array() * Z.array() * (1 - Z.array());
}