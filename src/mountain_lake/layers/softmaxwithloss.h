// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MOUNTAIN_LAKE_LAYERS_SOFTMAXWITHLOSS_H_
#define MOUNTAIN_LAKE_LAYERS_SOFTMAXWITHLOSS_H_

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXf;

void Softmax(Eigen::MatrixXf &A, Eigen::MatrixXf &Y);
float CrossEntropy(Eigen::MatrixXf &A, int t);

/// @brief Softmax与Loss合并层类（Softmax and Loss Merge Layer Classes）
class SoftmaxWithLoss {
 public:
  explicit SoftmaxWithLoss(){};
  ~SoftmaxWithLoss(){};
  float Forward(int label, MatrixXf &A, MatrixXf &Y);
  void Backward(MatrixXf &Y, uint8_t label, MatrixXf &dA);
};

#endif  // MOUNTAIN_LAKE_LAYERS_SOFTMAXWITHLOSS_H_
