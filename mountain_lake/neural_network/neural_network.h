// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef NEURAL_NETWORK_NEURAL_NETWORK_H_
#define NEURAL_NETWORK_NEURAL_NETWORK_H_

#include <mountain_town/string/toml.h>

#include <iostream>
#include <unordered_map>

using std::unordered_map;

/// @brief The structure of layers in neural networks
struct NeuralNetworkLayer {
  string type = "";       // 类型
  string name = "";       // 名称
  int output_height = 0;  // 输出矩阵的行数
  int output_width = 0;   // 输出矩阵的列数
  int output_size = 0;    // 输出数据的大小
};

// neural network class
class NeuralNetwork {
 public:
  NeuralNetwork();
  ~NeuralNetwork();
  string Init(string config_file);
  inline int GetLayersNum() { return this->layers_num_; }
  inline NeuralNetworkLayer& GetLayer(int index) { return this->nnl_[index]; }

 private:
  NeuralNetworkLayer nnl_[100];  // layers in neural networks
  int layers_num_;               // number of layers
};

#endif  // NEURAL_NETWORK_NEURAL_NETWORK_H_