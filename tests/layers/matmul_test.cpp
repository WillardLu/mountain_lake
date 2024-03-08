// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <mountain_lake/layers/matmul.h>

TEST(MatMulTests, Forward) {
  MatMul matmul;
  MatrixXf X = MatrixXf(1, 10);
  for (int i = 0; i < 10; ++i) {
    X(0, i) = i / 100.0f;
  }
  MatrixXf W = MatrixXf(10, 5);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 5; ++j) {
      W(i, j) = (i * 5 + j) / 100.0f;
    }
  }
  MatrixXf A = MatrixXf::Zero(1, 5);
  matmul.Forward(X, W, A);
  ASSERT_LT(abs(A(0, 0) - 0.1425f), 1e-7f);
  ASSERT_LT(abs(A(0, 4) - 0.1605f), 1e-7f);
  ASSERT_LT(abs(A(0, 2) - 0.1515f), 1e-7f);
}

TEST(MatMulTests, Backward) {
  MatMul matmul;
  MatrixXf X = MatrixXf(1, 10);
  for (int i = 0; i < 10; ++i) {
    X(0, i) = i / 100.0f;
  }
  MatrixXf W = MatrixXf(10, 5);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 5; ++j) {
      W(i, j) = (i * 5 + j) / 100.0f;
    }
  }
  MatrixXf dA = MatrixXf(1, 5);
  for (int i = 0; i < 5; ++i) {
    dA(0, i) = i / 100.0f;
  }
  MatrixXf dW = MatrixXf::Zero(10, 5);
  MatrixXf dX = MatrixXf::Zero(1, 10);

  matmul.Backward(X, W, dA, dW, dX, 3);
  ASSERT_LT(abs(dW(0, 0) - 0.0f), 1e-7);
  ASSERT_LT(abs(dW(9, 4) - 0.0036f), 1e-7);
  ASSERT_LT(abs(dW(2, 3) - 0.0006f), 1e-7);
  ASSERT_LT(abs(dX(0, 0) - 0.003f), 1e-7);
  ASSERT_LT(abs(dX(0, 4) - 0.023f), 1e-7);
  ASSERT_LT(abs(dX(0, 9) - 0.048f), 1e-7);
  dX = dX.setZero();
  matmul.Backward(X, W, dA, dW, dX, 1);
  ASSERT_LT(abs(dX(0, 4) - 0.0f), 1e-7);
}