// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MOUNTAIN_LAKE_NEURAL_NETWORK_NEURAL_NETWORK_H_
#define MOUNTAIN_LAKE_NEURAL_NETWORK_NEURAL_NETWORK_H_

#include <layers/affine.h>
#include <layers/convolution.h>
#include <layers/relu.h>
#include <layers/sigmoid.h>
#include <layers/softmaxwithloss.h>
#include <mountain_town/string/toml.h>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <unordered_map>

using Eigen::MatrixXf;
using std::unordered_map;

/// @brief 层的结构（structure of layers）
struct NeuralNetworkLayer {
  string type = "";
  string name = "";
  int output_height = 0;
  int output_width = 0;
  int output_size = 0;
};

/// @brief 原始数据结构（structure of raw data）
struct RowData {
  MatrixXf data = MatrixXf::Zero(1, 1);
  int height = 0;
  int width = 0;
  int size = 0;
};

/// @brief 神经网络类（neural network class）
class NeuralNetwork {
 public:
  NeuralNetwork();
  ~NeuralNetwork();
  string ReadConfig(string config_file);
  string Init(string config_file, RowData& raw_data);
  inline int GetLayers() { return this->layers_; }
  inline NeuralNetworkLayer& GetLayer(int index) { return this->nnl_[index]; }
  inline float GetLearningRate() { return this->learning_rate_; }

 private:
  string InitAffine(int i);
  void InitSigmoid(int i);
  void InitRelu(int i);
  void InitSoftmaxWithLoss(int i);
  string InitConv(unordered_map<string, string>& conf, int i);

  unordered_map<string, string> conf_;  // 配置信息（configuration information）
  NeuralNetworkLayer nnl_[100];         // 层（layers）
  int layers_;                          // 层的数量（number of layers）
  float learning_rate_;                 // 学习率（learning rate）
  uint8_t label_;                       // 监督标签

  MatrixXf* RD_;      // 原始数据（raw data）
  MatrixXf W_[100];   // 权重（weights）
  MatrixXf B_[100];   // 偏置（bias）
  MatrixXf O_[100];   // 层输出（output of layers）
  MatrixXf dW_[100];  // 权重的导数（derivative of weights)
  MatrixXf dB_[100];  // 偏置的导数（derivative of bias）
  MatrixXf dO_[100];  // 层输出参数的导数（The derivative of the output
                      // parameter of the layer）
  MatrixXf Y_;        // Softmax函数输出

  ConvConig cc_[100];  // 卷积层配置

  Affine affine_;
  Sigmoid sigmoid_;
  ReLU relu_;
  SoftmaxWithLoss softmax_loss_;
  Convolution conv_;
};

#endif  // MOUNTAIN_LAKE_NEURAL_NETWORK_NEURAL_NETWORK_H_