// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "neural_network.h"

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {}

/// @brief 读取配置信息（Read configuration information）
/// @param config_file 配置文件名称（Configuration file name）
/// @remark 配置文件要求是TOML文件。
///         The configuration file requirement is a TOML file.
/// @return 错误信息（error message）
string NeuralNetwork::ReadConfig(string config_file) {
  string err = ReadSTOML(config_file, this->conf_);
  if (!err.empty()) {
    return err;
  }
  vector<string> step_tmp;
  ReadSTOMLArr(this->conf_["neural_network.struct"], step_tmp);
  if (step_tmp.empty()) {
    return "undefined \"neural_network.struct\"";
  }
  vector<string> step_tmp1;
  vector<string> step_tmp2;
  int i = 0;
  for (auto &step : step_tmp) {
    ++i;
    this->nnl_[i].name = step;
    this->nnl_[i].type = step;
    // 检查是否设置了输出参数的大小。
    // Checks if the size of the output parameter is set.
    if (step.find(":") != string::npos) {
      step_tmp1 = SplitStr(step, ":");
      this->nnl_[i].output_size = atoi(step_tmp1[1].c_str());
      this->nnl_[i].name = step_tmp1[0];
      this->nnl_[i].type = step_tmp1[0];
    }
    // 检查是否需要补充定义
    // Check if the entry needs additional definitions.
    if (this->nnl_[i].name.find("-") != string::npos) {
      step_tmp2 = SplitStr(this->nnl_[i].name, "-");
      this->nnl_[i].type = step_tmp2[0];
    }
  }
  this->layers_ = i;
  return "";
}

/// @brief 初始化神经网络（Initialize the neural network）
/// @param config_file 配置文件名称（Configuration file name）
/// @remark 配置文件要求是TOML文件。
///         The configuration file requirement is a TOML file.
/// @return 错误信息（error message）
string NeuralNetwork::Init(string config_file) {
  string err = this->ReadConfig(config_file);
  if (!err.empty()) {
    return err;
  }
  this->learning_rate_ = stof(this->conf_["hyper_parameters.learning_rate"]);
  // 逐层进行初始化（Layer-by-layer initialization）
  for (int i = 1; i <= this->layers_; ++i) {
    // 初始化仿射变换层（Initialize the affine transformation layer）
    if (this->nnl_[i].type == "Affine") {
      err = this->InitAffine(i);
      if (err.empty() == false) {
        return err;
      }
      continue;
    }
  }
  return "";
}

/// @brief 初始化仿射变换层（Initialize the affine transformation layer）
/// @param i 当前层号（current layer number）
/// @return 错误信息（error message）
string NeuralNetwork::InitAffine(int i) {
  this->nnl_[i].output_height = 1;
  this->nnl_[i].output_width = this->nnl_[i].output_size;
  int size = this->nnl_[i - 1].output_size * this->nnl_[i].output_size;
  this->W_[i] =
      MatrixXf(this->nnl_[i - 1].output_size, this->nnl_[i].output_size);
  float *w_tmp = new float[size];
  if (w_tmp == nullptr) {
    return "Memory allocation failed when initializing the affine "
           "transformation layer.";
  }
  NormalDistr(w_tmp, size, 0.0, 0.01);
  memcpy(this->W_[i].data(), w_tmp, size * sizeof(float));
  delete[] w_tmp;
  this->dW_[i] =
      MatrixXf::Zero(this->nnl_[i - 1].output_size, this->nnl_[i].output_size);
  this->B_[i] = MatrixXf::Zero(1, this->nnl_[i].output_size);
  this->dB_[i] = MatrixXf::Zero(1, this->nnl_[i].output_size);
  this->O_[i] = MatrixXf::Zero(1, this->nnl_[i].output_size);
  this->dO_[i] = MatrixXf::Zero(1, this->nnl_[i].output_size);
  return "";
}