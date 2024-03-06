// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "neural_network.h"

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {}

/// @brief Initialize the neural network
/// @param config_file Configuration file name
/// @remark The configuration file requirement is a TOML file.
/// @return error message
string NeuralNetwork::Init(string config_file) {
  unordered_map<string, string> conf;
  string err = ReadSTOML(config_file, conf);
  if (err.empty() == false) {
    return err;
  }
  vector<string> step_tmp;
  ReadSTOMLArr(conf["neural_network.struct"], step_tmp);
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
    // Check if the size of the output parameter is defined
    if (step.find(":") != string::npos) {
      step_tmp1 = SplitStr(step, ":");
      this->nnl_[i].output_size = atoi(step_tmp1[1].c_str());
      this->nnl_[i].name = step_tmp1[0];
      this->nnl_[i].type = step_tmp1[0];
    }
    // Check if the entry needs additional definitions.
    if (this->nnl_[i].name.find("-") != string::npos) {
      step_tmp2 = SplitStr(this->nnl_[i].name, "-");
      this->nnl_[i].type = step_tmp2[0];
    }
  }
  this->layers_num_ = i;
  return "";
}