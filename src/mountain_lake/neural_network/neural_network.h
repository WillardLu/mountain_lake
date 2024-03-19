// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MOUNTAIN_LAKE_NEURAL_NETWORK_NEURAL_NETWORK_H_
#define MOUNTAIN_LAKE_NEURAL_NETWORK_NEURAL_NETWORK_H_

#include <mountain_lake/layers/affine.h>
#include <mountain_lake/layers/convolution.h>
#include <mountain_lake/layers/matmul.h>
#include <mountain_lake/layers/pooling.h>
#include <mountain_lake/layers/relu.h>
#include <mountain_lake/layers/sigmoid.h>
#include <mountain_lake/layers/softmaxwithloss.h>
#include <mountain_town/string/toml.h>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <unordered_map>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::RowMajor;
using std::unordered_map;

typedef Matrix<uint8_t, Dynamic, Dynamic> MatrixXb;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfr;

/// @brief 层的结构（structure of layers）
struct NeuralNetworkLayer {
  string type = "";
  string name = "";
  int output_height = 0;
  int output_width = 0;
  int output_size = 0;
};

/// @brief 原始数据结构（structure of raw data）
struct RawData {
  MatrixXfr train_data = MatrixXfr::Zero(1, 1);
  MatrixXb train_labels = MatrixXb::Zero(1, 1);
  MatrixXfr test_data = MatrixXfr::Zero(1, 1);
  MatrixXb test_labels = MatrixXb::Zero(1, 1);
  // 单个数据矩阵的行数（Number of rows in a single data matrix）
  int row = 0;
  // 单个数据矩阵的列数（Number of columns in a single data matrix）
  int col = 0;
  // 单个数据的大小（Size of individual data）
  int size = 0;
  int train_number = 0;
  int test_number = 0;
};

/// @brief 神经网络类（neural network class）
class NeuralNetwork {
 public:
  NeuralNetwork();
  ~NeuralNetwork();
  string ReadConfig(string config_file);
  string Init(string config_file, RawData& train_data);
  inline int GetLayers() { return this->layers_; }
  inline NeuralNetworkLayer& GetLayer(int index) { return this->nnl_[index]; }
  inline float GetLearningRate() { return this->learning_rate_; }
  inline void SetLearningRate(float rate) { this->learning_rate_ = rate; }
  inline RawData& GetTrainData() { return this->raw_data_; }
  void Gradient(int index);
  void Forward();
  void Predict();
  void Backward();
  void Update();
  void Accuracy(string& csv);

 private:
  string InitAffine(int i);
  void InitSigmoid(int i);
  void InitRelu(int i);
  void InitSoftmaxWithLoss(int i);
  string InitConv(int i);
  string InitPool(int i);

  unordered_map<string, string> conf_;  // 配置信息（configuration information）
  NeuralNetworkLayer nnl_[100];         // 层（layers）
  int layers_;                          // 层的数量（number of layers）
  float learning_rate_;                 // 学习率（learning rate）
  uint8_t label_;                       // 监督标签

  RawData raw_data_;  // 原始数据（raw data）
  MatrixXf W_[100];   // 权重（weights）
  MatrixXf B_[100];   // 偏置（bias）
  MatrixXf O_[100];   // 层输出（output of layers）
  MatrixXf dW_[100];  // 权重的导数（derivative of weights)
  MatrixXf dB_[100];  // 偏置的导数（derivative of bias）
  MatrixXf dO_[100];  // 层输出参数的导数（The derivative of the output
                      // parameter of the layer）
  MatrixXf Y_;        // Softmax函数输出（Softmax function output）
  float loss_;        // 误差（error）

  ConvConig cc_[100];  // 卷积层配置（Convolutional Layer Configuration）
  PoolConfig pc_[100];  // 池化层配置（Pooling layer configuration）

  Affine affine_;
  Sigmoid sigmoid_;
  ReLU relu_;
  SoftmaxWithLoss softmax_loss_;
  Convolution conv_;
  Pooling pool_;
  MatMul matmul_;
};

#endif  // MOUNTAIN_LAKE_NEURAL_NETWORK_NEURAL_NETWORK_H_