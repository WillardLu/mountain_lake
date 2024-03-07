// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "relu.h"

/// @brief 线性整流层正向传播
/// @param A 输入信号
/// @param Z 输出信号
void ReLU::Forward(MatrixXf &A, MatrixXf &Z) {
  // 如果A的值大于0，Z就等于A，否则Z等于0
  Z = (A.array() > 0).select(A, 0);
}

/// @brief 线性整流层反向传播
/// @param dZ 输出信号的导数
/// @param Z 输出信号
/// @param dA 输入信号的导数
void ReLU::Backward(MatrixXf &dZ, MatrixXf &Z, MatrixXf &dA) {
  dA = (Z.array() > 0).select(dZ, 0);
}