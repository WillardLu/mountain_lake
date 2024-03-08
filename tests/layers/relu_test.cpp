// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <mountain_lake/layers/relu.h>

TEST(ReluTests, Forward) {
  ReLU relu;
  MatrixXf A = MatrixXf(1, 5);
  for (int i = 0; i < 5; ++i) {
    A(0, i) = i / 100.0f;
  }
  MatrixXf Z = MatrixXf::Zero(1, 5);
  relu.Forward(A, Z);
  ASSERT_EQ(Z(0, 0), 0);
  ASSERT_EQ(Z(0, 2), 0.02f);
}

TEST(ReluTests, Backward) {
  ReLU relu;
  MatrixXf dZ = MatrixXf::Zero(1, 5);
  for (int i = 0; i < 5; ++i) {
    dZ(0, i) = i / 100.0f;
  }
  MatrixXf Z = MatrixXf(1, 5);
  for (int i = 0; i < 5; ++i) {
    Z(0, i) = i / 50.0f;
  }
  MatrixXf dA = MatrixXf::Zero(1, 5);
  relu.Backward(dZ, Z, dA);
  ASSERT_EQ(dA(0, 0), 0);
  ASSERT_EQ(dA(0, 2), 0.02f);
}