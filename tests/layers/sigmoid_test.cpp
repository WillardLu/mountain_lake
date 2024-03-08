// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <mountain_lake/layers/sigmoid.h>

TEST(SigmoidTests, Forward) {
  Sigmoid sigmoid;
  MatrixXf A = MatrixXf(1, 5);
  for (int i = 0; i < 5; ++i) {
    A(0, i) = i / 100.0f;
  }
  MatrixXf Z = MatrixXf::Zero(1, 5);
  sigmoid.Forward(A, Z);
  ASSERT_LT(abs(Z(0, 0) - 0.5f), 1e-7);
  ASSERT_LT(abs(Z(0, 2) - 0.50499983f), 1e-7);
}

TEST(SigmoidTests, Backward) {
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
  ASSERT_LT(abs(dA(0, 2) - 0.000768f), 1e-7);
}