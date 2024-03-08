// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <mountain_lake/neural_network/neural_network.h>

TEST(NNTest, ReadConfig) {
  NeuralNetwork nn;
  string err = nn.ReadConfig("tests/testdata/config.toml");
  ASSERT_EQ(err, "");
  ASSERT_EQ(nn.GetLayers(), 4);
  ASSERT_EQ(nn.GetLayer(2).name, "Sigmoid");
  ASSERT_EQ(nn.GetLayer(3).type, "Affine");
}

TEST(NNTest, Init) {
  NeuralNetwork nn;
  RawData raw_data;
  raw_data.train_data = MatrixXfr::Random(100, 784);
  raw_data.train_labels = MatrixXb::Random(100, 1);
  raw_data.row = 28;
  raw_data.col = 28;
  string err = nn.Init("tests/testdata/config.toml", raw_data);
  ASSERT_EQ(err, "");
  ASSERT_LT(nn.GetLearningRate() - 0.01, 1e-7);
  ASSERT_EQ(nn.GetTrainData().train_data.rows(), 100);
}