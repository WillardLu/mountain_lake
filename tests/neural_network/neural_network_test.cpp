// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <neural_network/neural_network.h>

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
  string err = nn.Init("tests/testdata/config.toml");
  ASSERT_EQ(err, "");
  ASSERT_EQ(nn.GetLearningRate() * 100, 1);
}