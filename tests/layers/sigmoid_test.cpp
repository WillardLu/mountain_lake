// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <layers/sigmoid.h>

TEST(SigmoidLayer, Forward) {
  Sigmoid sigmoid;
  MatrixXf A = MatrixXf(1, 5);
  for (int i = 0; i < 5; ++i) {
    A(0, i) = i / 100.0f;
  }
  MatrixXf Z = MatrixXf::Zero(1, 5);
  sigmoid.Forward(A, Z);
  ASSERT_EQ(Z(0, 0), 0.5f);
  ASSERT_EQ(round(Z(0, 2) * 10000), 5050);
}

TEST(SigmoidLayer, Backward) {
  Sigmoid sigmoid;
  MatrixXf dZ = MatrixXf::Zero(1, 5);
  for (int i = 0; i < 5; ++i) {
    dZ(0, i) = i / 100.0f;
  }
  MatrixXf Z = MatrixXf(1, 5);
  for (int i = 0; i < 5; ++i) {
    Z(0, i) = i / 50.0f;
  }
  MatrixXf dA = MatrixXf::Zero(1, 5);
  sigmoid.Backward(dZ, Z, dA);
  ASSERT_EQ(dA(0, 0), 0);
  ASSERT_EQ(round(dA(0, 2) * 1e6), 768);
}