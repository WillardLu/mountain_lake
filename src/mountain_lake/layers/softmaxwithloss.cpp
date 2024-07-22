// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "softmaxwithloss.h"

/// @brief Softmax函数（Softmax function）
/// @param A 输入（input）
/// @param Y 输出（output）
void Softmax(Eigen::MatrixXf &A, Eigen::MatrixXf &Y) {
  Eigen::MatrixXf X_exp = (A.array() - A.maxCoeff()).array().exp();
  Y = X_exp / X_exp.sum();
}

/// @brief 交叉熵函数（cross-entropy function）
/// @param Y 输入（input）
/// @param t 监督标签（Supervisory label）
/// @return 交叉熵误差（cross-entropy error）
float CrossEntropy(Eigen::MatrixXf &Y, int t) {
  return -std::log(Y(0, t) + 1e-7);
}

/// @brief softmax与交叉熵误差合并层的正向传播
/// Forward propagation of softmax and cross-entropy error merger layers
/// @param labels 监督标签（Supervisory label）
/// @param A 输入（input）
/// @param Y 输出（output）
/// @return 误差（errors）
float SoftmaxWithLoss::Forward(int label, MatrixXf &A, MatrixXf &Y) {
  Softmax(A, Y);
  return CrossEntropy(Y, label);
}

/// @brief softmax与交叉熵误差合并层的反向传播
/// Backpropagation of softmax and cross-entropy error merger layers
/// @param Y 经过softmax函数处理过的信号（Signal processed by softmax function）
/// @param labels 监督标签（Supervisory label）
/// @param dA 输入信号的导数（Derivative of the input signal）
void SoftmaxWithLoss::Backward(MatrixXf &Y, uint8_t label, MatrixXf &dA) {
  Y(0, label) -= 1;
  dA = Y;
}