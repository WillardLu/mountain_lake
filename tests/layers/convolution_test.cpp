// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <mountain_lake/layers/convolution.h>

/// @brief 卷积层测试
TEST(ConvolutionTests, Composite) {
  // 准备测试数据
  // 1. 准备图像数据
  MatrixXf X = MatrixXf(1, 784);
  for (int i = 0; i < 784; ++i) {
    X(0, i) = i / 1000.0f;
  }
  // 2. 准备配置数据
  ConvConig cc[2];
  cc[1].stride = 1;
  cc[1].height = 5;
  cc[1].width = 5;
  cc[1].number = 3;
  cc[1].i_height = 28;
  cc[1].i_width = 28;
  cc[1].o_height =
      (cc[1].i_height - cc[1].height + 2 * cc[1].pad) / cc[1].stride + 1;
  cc[1].o_width =
      (cc[1].i_width - cc[1].width + 2 * cc[1].pad) / cc[1].stride + 1;
  // 3. 准备权重W的数据
  MatrixXf W[10];
  W[1] = MatrixXf(1, cc[1].number * cc[1].height * cc[1].width);
  for (int i = 0; i < cc[1].number * cc[1].height * cc[1].width; ++i) {
    W[1](0, i) = i / 100.0f;
  }
  // 4. 准备偏置B的数据
  MatrixXf B[10];
  B[1] = MatrixXf(1, cc[1].number);
  for (int i = 0; i < cc[1].number; ++i) {
    B[1](0, i) = i / 10.0f + 0.05;
  }
  // 5. 准备输出数据O
  MatrixXf O[10];
  O[1] = MatrixXf::Zero(1, cc[1].number * cc[1].o_height * cc[1].o_width);

  Convolution conv;
  conv.Forward(X, W[1], B[1], O[1], cc[1]);

  // 普通测试1
  MatrixXf X_2d = X.reshaped<RowMajor>(cc[1].i_height, cc[1].i_width);
  float sum1 = 0.0f;
  for (int i = 0; i < cc[1].height; ++i) {
    for (int j = 0; j < cc[1].width; ++j) {
      sum1 += W[1](0, i * cc[1].width + j) * X_2d(i, j);
    }
  }
  sum1 += B[1](0, 0);
  ASSERT_LT(abs(O[1](0, 0) - sum1), 1e-8);
  // 普通测试2
  sum1 = 0.0f;
  for (int i = 0; i < cc[1].height; ++i) {
    for (int j = 0; j < cc[1].width; ++j) {
      sum1 += W[1](0, i * cc[1].width + j) * X_2d(i, j + 5);
    }
  }
  sum1 += B[1](0, 0);
  ASSERT_LT(abs(O[1](0, 5) - sum1), 1e-8);
  // 普通测试3
  sum1 = 0.0f;
  for (int i = 0; i < cc[1].height; ++i) {
    for (int j = 0; j < cc[1].width; ++j) {
      sum1 += W[1](0, i * cc[1].width + j) * X_2d(i + 23, j + 23);
    }
  }
  sum1 += B[1](0, 0);
  ASSERT_LT(abs(O[1](0, 23 * 24 + 23) - sum1), 1e-8);

  // 准备偏置导数
  MatrixXf dB[10];
  dB[1] = MatrixXf::Zero(1, cc[1].number);
  // 准备输出参数导数
  MatrixXf dO[10];
  dO[1] = MatrixXf(1, cc[1].number * cc[1].o_height * cc[1].o_width);
  for (int i = 0; i < cc[1].number * cc[1].o_height * cc[1].o_width; ++i) {
    dO[1](0, i) = i / 2000.0f;
  }
  // 准备权重导数
  MatrixXf dW[10];
  dW[1] = MatrixXf::Zero(1, cc[1].number * cc[1].height * cc[1].width);

  conv.Backward(X, dB[1], dO[1], dW[1], cc[1], 1);
  int size2 = cc[1].o_height * cc[1].o_width;
  MatrixXf dO_tmp = dO[1].reshaped<RowMajor>(cc[1].number, size2);

  // 普通测试4
  ASSERT_EQ(dB[1](0, 0), dO_tmp.row(0).sum());
  // 普通测试5。因为这里的步幅为1，所以可以直接用这种方式取值。
  sum1 = (X_2d.block(0, 0, cc[1].o_height, cc[1].o_width)
              .reshaped<RowMajor>(1, cc[1].o_height * cc[1].o_width) *
          dO[1].block(0, 0, 1, size2).reshaped<RowMajor>(size2, 1))(0, 0);
  ASSERT_EQ(dW[1](0, 0), sum1);
  // 普通测试6
  sum1 = (X_2d.block(0, 3, cc[1].o_height, cc[1].o_width)
              .reshaped<RowMajor>(1, cc[1].o_height * cc[1].o_width) *
          dO[1].block(0, 0, 1, size2).reshaped<RowMajor>(size2, 1))(0, 0);
  ASSERT_EQ(dW[1](0, 3), sum1);
}