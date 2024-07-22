// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include <gtest/gtest.h>
#include <mountain_lake/layers/pooling.h>

/// @brief 池化层测试
TEST(PoolingTests, Composite) {
  PoolConfig pc[2];
  pc[1].height = 2;
  pc[1].width = 2;
  pc[1].stride = 2;
  pc[1].filter_num = 4;
  pc[1].type = 0;
  pc[1].i_height = 24;
  pc[1].i_width = 24;
  pc[1].o_height = (pc[1].i_height - pc[1].height) / pc[1].stride + 1;
  pc[1].o_width = (pc[1].i_width - pc[1].width) / pc[1].stride + 1;
  Pooling pool;
  // 准备输出参数导数
  MatrixXf A[10];
  MatrixXf dA[10];
  A[1] = MatrixXf(1, pc[1].filter_num * pc[1].i_height * pc[1].i_width);
  dA[1] = MatrixXf(1, pc[1].filter_num * pc[1].i_height * pc[1].i_width);
  for (int i = 0; i < pc[1].filter_num * pc[1].o_height * pc[1].o_width; ++i) {
    A[1](0, i) = i / 2000.0f;
  }
  // 准备输出参数导数
  MatrixXf O[10];
  O[1] = MatrixXf(1, pc[1].filter_num * pc[1].o_height * pc[1].o_width);
  // ------正向传播------
  pool.Forward(A[1], O[1], pc[1]);

  // int size1 = pc[1].height * pc[1].width;
  int size2 = pc[1].i_height * pc[1].i_width;
  MatrixXf A_tmp = MatrixXf(pc[1].i_height, pc[1].i_width);
  A_tmp = A[1].block(0, 0, 1, size2)
              .reshaped<RowMajor>(pc[1].i_height, pc[1].i_width);
  MatrixXf data1 = MatrixXf(pc[1].height, pc[1].width);
  data1 = A_tmp.block(0, 0, pc[1].height, pc[1].width);
  float sum1 = data1.maxCoeff();
  // 普通测试1
  ASSERT_EQ(O[1](0, 0), sum1);
  data1 = A_tmp.block(0, 2, pc[1].height, pc[1].width);
  sum1 = data1.maxCoeff();
  // 普通测试2
  ASSERT_EQ(O[1](0, 1), sum1);
  data1 = A_tmp.block(2, 2, pc[1].height, pc[1].width);
  sum1 = data1.maxCoeff();
  // 普通测试3
  ASSERT_EQ(O[1](0, 13), sum1);

  MatrixXf dZ[10];
  dZ[1] = MatrixXf(1, pc[1].filter_num * pc[1].o_height * pc[1].o_width);
  for (int i = 0; i < pc[1].filter_num * pc[1].o_height * pc[1].o_width; ++i) {
    dZ[1](0, i) = i / 2000.0f;
  }

  // ------反向传播------
  pool.Backward(dZ[1], dA[1], A[1], pc[1]);
  // 普通测试1
  ASSERT_EQ(dA[1](0, 0), 0);
  // 普通测试2
  ASSERT_LT(abs(dA[1](0, 73) - 0.006), 1e-8);
}