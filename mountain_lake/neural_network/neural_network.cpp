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
/// @param raw_data 原始数据（raw data）
/// @remark 配置文件要求是TOML文件。
///         The configuration file requirement is a TOML file.
/// @return 错误信息（error message）
string NeuralNetwork::Init(string config_file, RowData &raw_data) {
  string err = this->ReadConfig(config_file);
  if (!err.empty()) {
    return err;
  }
  this->learning_rate_ = stof(this->conf_["hyper_parameters.learning_rate"]);
  // 逐层进行初始化（Layer-by-layer initialization）
  this->RD_ = &raw_data.data;
  this->nnl_[0].output_height = raw_data.height;
  this->nnl_[0].output_width = raw_data.width;
  this->nnl_[0].output_size = raw_data.size;
  for (int i = 1; i <= this->layers_; ++i) {
    // 初始化仿射变换层（Initialize the affine transformation layer）
    if (this->nnl_[i].type == "Affine") {
      err = this->InitAffine(i);
      if (err.empty() == false) {
        return err;
      }
      continue;
    }
    // 初始化Sigmoid层参数
    if (this->nnl_[i].type == "Sigmoid") {
      this->InitSigmoid(i);
      continue;
    }
    // 初始化线性整流层参数
    if (this->nnl_[i].type == "ReLU") {
      this->InitRelu(i);
      continue;
    }
    // 初始化SoftmaxWithLoss层参数
    if (this->nnl_[i].type == "SoftmaxWithLoss") {
      this->InitSoftmaxWithLoss(i);
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

/// @brief 初始化sigmoid激活函数层
/// @param i 序号
void NeuralNetwork::InitSigmoid(int i) {
  this->nnl_[i].output_height = this->nnl_[i - 1].output_height;
  this->nnl_[i].output_width = this->nnl_[i - 1].output_width;
  this->nnl_[i].output_size = this->nnl_[i - 1].output_size;
  this->O_[i] = MatrixXf::Zero(1, this->nnl_[i].output_size);
  this->dO_[i] = MatrixXf::Zero(1, this->nnl_[i].output_size);
}

/// @brief 初始化线性整流激活函数层
/// @param i 序号
void NeuralNetwork::InitRelu(int i) {
  this->nnl_[i].output_height = this->nnl_[i - 1].output_height;
  this->nnl_[i].output_width = this->nnl_[i - 1].output_width;
  this->nnl_[i].output_size = this->nnl_[i - 1].output_size;
  this->O_[i] = MatrixXf::Zero(1, this->nnl_[i].output_size);
  this->dO_[i] = MatrixXf::Zero(1, this->nnl_[i].output_size);
}

/// @brief 初始化SoftmaxWithLoss层
/// @param i 序号
void NeuralNetwork::InitSoftmaxWithLoss(int i) {
  this->nnl_[i].output_height = 1;
  this->nnl_[i].output_width = 1;
  this->nnl_[i].output_size = 1;
  this->O_[i] = MatrixXf::Zero(1, 1);
  this->dO_[i] = MatrixXf::Zero(1, 1);
  this->Y_ = MatrixXf::Zero(1, this->nnl_[i - 1].output_size);
}

/// @brief 初始化卷积层
/// @param conf 配置内容
/// @param i 序号
/// @return 错误信息
string NeuralNetwork::InitConv(unordered_map<string, string> &conf, int i) {
  if (conf[this->nnl_[i].name + ".pad"].empty() == true ||
      conf[this->nnl_[i].name + ".stride"].empty() == true ||
      conf[this->nnl_[i].name + ".filter_num"].empty() == true ||
      conf[this->nnl_[i].name + ".filter_height"].empty() == true ||
      conf[this->nnl_[i].name + ".filter_width"].empty() == true ||
      conf[this->nnl_[i].name + ".channel_num"].empty() == true) {
    return "错误：“" + this->nnl_[i].name + "”内容不全，请检查配置文件。\n";
  }
  int pad = atoi(conf[this->nnl_[i].name + ".pad"].c_str());
  int stride = atoi(conf[this->nnl_[i].name + ".stride"].c_str());
  int f_num = atoi(conf[this->nnl_[i].name + ".filter_num"].c_str());
  int f_height = atoi(conf[this->nnl_[i].name + ".filter_height"].c_str());
  int f_width = atoi(conf[this->nnl_[i].name + ".filter_width"].c_str());
  int channel_num = atoi(conf[this->nnl_[i].name + ".channel_num"].c_str());
  int o_height =
      (this->nnl_[i - 1].output_height - f_height + 2 * pad) / stride + 1;
  int o_width =
      (this->nnl_[i - 1].output_width - f_width + 2 * pad) / stride + 1;
  this->nnl_[i].output_height = o_height;
  this->nnl_[i].output_width = o_width;
  this->nnl_[i].output_size = o_height * o_width * f_num;
  this->cc_[i].pad = pad;
  this->cc_[i].stride = stride;
  this->cc_[i].number = f_num;
  this->cc_[i].height = f_height;
  this->cc_[i].width = f_width;
  this->cc_[i].channel_num = channel_num;
  this->cc_[i].i_height = this->nnl_[i - 1].output_height;
  this->cc_[i].i_width = this->nnl_[i - 1].output_width;
  this->cc_[i].o_height = o_height;
  this->cc_[i].o_width = o_width;
  // 初始化权重
  this->W_[i] = MatrixXf(1, f_num * f_height * f_width);
  float *w_tmp = new float[f_num * f_height * f_width];
  if (w_tmp == nullptr) {
    return "初始化卷积层时内存分配失败。\n";
  }
  // 这里的初始化参数设置很重要
  NormalDistr(w_tmp, f_num * f_height * f_width, 0.0, 0.01);
  memcpy(this->W_[i].data(), w_tmp, f_num * f_height * f_width * sizeof(float));
  delete[] w_tmp;
  // 初始化权重的导数
  this->dW_[i] = MatrixXf::Zero(1, f_num * f_height * f_width);
  // 初始化偏置
  this->B_[i] = MatrixXf::Zero(1, f_num);
  // 初始化偏置的导数
  this->dB_[i] = MatrixXf::Zero(1, f_num);
  // 初始化输出参数
  this->O_[i] = MatrixXf::Zero(1, o_height * o_width * f_num);
  this->dO_[i] = MatrixXf::Zero(1, o_height * o_width * f_num);
  return "";
}