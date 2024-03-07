// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <layers/affine.h>

TEST(AffineLayer, Forward) {
  Affine affine;
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
  MatrixXf B = MatrixXf(1, 5);
  for (int i = 0; i < 5; ++i) {
    B(0, i) = i / 100.0f;
  }
  MatrixXf A = MatrixXf::Zero(1, 5);
  affine.Forward(X, W, B, A);
  ASSERT_LT(A(0, 0) - 0.1425f, 1e-7f);
  ASSERT_LT(A(0, 4) - 0.2005f, 1e-7f);
  ASSERT_LT(A(0, 2) - 0.1715f, 1e-7f);
}

TEST(AffineLayer, Backward) {
  Affine affine;
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
  MatrixXf dB = MatrixXf::Zero(1, 5);
  MatrixXf dW = MatrixXf::Zero(10, 5);
  MatrixXf dX = MatrixXf::Zero(1, 10);

  affine.Backward(X, W, dA, dB, dW, dX, 3);
  ASSERT_EQ(dB(0, 3), dA(0, 3));
  ASSERT_LT(dW(0, 0) - 0.0f, 1e-7);
  ASSERT_LT(dW(9, 4) - 36.0f, 1e-7);
  ASSERT_LT(dW(2, 3) - 6.0f, 1e-7);
  ASSERT_LT(dX(0, 0) - 30.0f, 1e-7);
  ASSERT_LT(dX(0, 4) - 230.0f, 1e-7);
  ASSERT_LT(dX(0, 9) - 480.0f, 1e-7);
  dX = dX.setZero();
  affine.Backward(X, W, dA, dB, dW, dX, 1);
  ASSERT_LT(dX(0, 4) - 0.0f, 1e-7);
}