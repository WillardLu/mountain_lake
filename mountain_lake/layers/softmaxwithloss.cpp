// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "softmaxwithloss.h"

/// @param A 输入矩阵
/// @param Y 输出矩阵
void Softmax(Eigen::MatrixXf &A, Eigen::MatrixXf &Y) {
  Eigen::MatrixXf X_exp = (A.array() - A.maxCoeff()).array().exp();
  Y = X_exp / X_exp.sum();
}

/// @brief 交叉熵函数
/// @param Y 输入矩阵
/// @param t 标签
/// @return 交叉熵误差
float CrossEntropy(Eigen::MatrixXf &Y, int t) {
  return -std::log(Y(0, t) + 1e-7);
}

/// @brief softmax与交叉熵误差合并层的正向传播
/// @param labels 监督标签
/// @param A 输入信号
/// @param Y 输出信号
/// @return 误差
float SoftmaxWithLoss::Forward(int label, MatrixXf &A, MatrixXf &Y) {
  Softmax(A, Y);
  return CrossEntropy(Y, label);
}

/// @brief softmax与交叉熵误差合并层的反向传播
/// @param Y 经过softmax函数处理过的信号
/// @param labels 监督标签
/// @param dA 输入信号的导数
void SoftmaxWithLoss::Backward(MatrixXf &Y, uint8_t label, MatrixXf &dA) {
  Y(0, label) -= 1;
  dA = Y;
}