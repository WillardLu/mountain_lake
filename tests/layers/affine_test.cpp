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
  ASSERT_EQ(A(0, 0), 0.1425f);
  ASSERT_EQ(round(A(0, 4) * 10000), 2005);
  ASSERT_EQ(A(0, 2), 0.1715f);
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
  ASSERT_EQ(dW(0, 0), 0.0f);
  ASSERT_EQ(round(dW(9, 4) * 10000), 36);
  ASSERT_EQ(round(dW(2, 3) * 10000), 6);
  ASSERT_EQ(round(dX(0, 0) * 10000), 30);
  ASSERT_EQ(round(dX(0, 4) * 10000), 230);
  ASSERT_EQ(round(dX(0, 9) * 10000), 480);
  dX = dX.setZero();
  affine.Backward(X, W, dA, dB, dW, dX, 1);
  ASSERT_EQ(dX(0, 4), 0);
}