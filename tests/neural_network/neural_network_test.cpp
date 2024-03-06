// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <neural_network/neural_network.h>

TEST(NNTest, Init) {
  NeuralNetwork nn;
  string err = nn.Init("tests/testdata/config.toml");
  ASSERT_EQ(err, "");
  ASSERT_EQ(nn.GetLayersNum(), 7);
  ASSERT_EQ(nn.GetLayer(2).name, "ReLU");
  ASSERT_EQ(nn.GetLayer(3).type, "Pooling");
}