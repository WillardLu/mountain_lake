// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <mountain_lake/layers/softmaxwithloss.h>

TEST(SoftmaxWithLossTests, Forward) {
  SoftmaxWithLoss softmax_loss;
  int label = 3;
  MatrixXf A = MatrixXf(1, 10);
  for (int i = 0; i < 10; ++i) {
    A(0, i) = i / 100.0f;
  }
  MatrixXf Y = MatrixXf(1, 10);
  float loss = softmax_loss.Forward(label, A, Y);
  ASSERT_LT(abs(Y(0, 0) - 0.09556032f), 1e-8);
  ASSERT_LT(abs(Y(0, 5) - 0.10045981f), 1e-8);
  ASSERT_LT(abs(Y(0, 9) - 0.10455965f), 1e-8);
  ASSERT_LT(abs(loss - 2.317996542749472f), 1e-8);
}

TEST(SoftmaxWithLossTests, Backward) {
  SoftmaxWithLoss softmax_loss;
  int label = 3;
  MatrixXf Y = MatrixXf(1, 10);
  for (int i = 0; i < 10; ++i) {
    Y(0, i) = i / 100.0f;
  }
  MatrixXf dA = MatrixXf(1, 10);
  softmax_loss.Backward(Y, label, dA);
  ASSERT_LT(abs(dA(0, 0) - 0.0f), 1e-8);
  ASSERT_LT(abs(dA(0, 3) + 0.97f), 1e-8);
  ASSERT_LT(abs(dA(0, 9) - 0.09f), 1e-8);
}