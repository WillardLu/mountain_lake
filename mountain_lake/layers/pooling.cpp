// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "pooling.h"

/// @brief 池化层正向传播（Pooling layer forward propagation）
/// @param A 输入（input）
/// @param O 输出（output）
/// @param pc 配置内容（Configuration contents）
void Pooling::Forward(MatrixXf &A, MatrixXf &O, PoolConfig &pc) {
  int size1 = pc.height * pc.width;
  int size2 = pc.i_height * pc.i_width;
  MatrixXf data1 = MatrixXf(pc.height, pc.width);
  MatrixXf A_tmp = MatrixXf(pc.i_height, pc.i_width);
  int k = -1;
  for (int m = 0; m < pc.filter_num; ++m) {
    for (int i = 0; i < size2; ++i) {
      A_tmp(i / pc.i_width, i % pc.i_width) = A(0, m * size2 + i);
    }
    for (int i = 0; i < pc.i_height; i += pc.stride) {
      for (int j = 0; j < pc.i_width; j += pc.stride) {
        for (int k = 0; k < pc.height; ++k) {
          for (int l = 0; l < pc.width; ++l) {
            data1(k, l) = A_tmp(i + k, j + l);
          }
        }
        ++k;
        if (pc.type == 0) {
          // 取最大值（take the maximum value）
          O(0, k) = data1.maxCoeff();
          continue;
        }
        if (pc.type == 1) {
          // 取平均值（take an average value）
          O(0, k) = data1.sum() / size1;
          continue;
        }
      }
    }
  }
}

/// @brief 池化层反向传播（Pooling layer backpropagation）
/// @param dZ 输出参数的导数（Derivatives of output parameters）
/// @param Z 输出（output）
/// @param dA 输入参数的导数（Derivatives of input parameters）
/// @param A 输入（input）
/// @param pc 配置内容（Configuration contents）
void Pooling::Backward(MatrixXf &dZ, MatrixXf &dA, MatrixXf &A,
                       PoolConfig &pc) {
  int size2 = pc.i_height * pc.i_width;
  int max_row = 0;
  int max_col = 0;
  MatrixXf data1 = MatrixXf(pc.height, pc.width);
  MatrixXf A_tmp = MatrixXf(pc.i_height, pc.i_width);
  MatrixXf dA_tmp = MatrixXf(pc.i_height, pc.i_width);
  for (int m = 0; m < pc.filter_num; ++m) {
    A_tmp = A.block(0, m * size2, 1, size2)
                .reshaped<RowMajor>(pc.i_height, pc.i_width);
    for (int i = 0; i < pc.o_height; ++i) {
      for (int j = 0; j < pc.o_width; ++j) {
        data1 = A_tmp.block(i * pc.stride, j * pc.stride, pc.height, pc.width);
        if (pc.type == 0) {
          // 取最大值情况下的反向求导
          // Backward derivation in the case of maxima
          data1.maxCoeff(&max_row, &max_col);
          for (int k = 0; k < pc.height; ++k) {
            for (int l = 0; l < pc.width; ++l) {
              data1(k, l) =
                  k == max_row && l == max_col
                      ? dZ(0, m * pc.o_height * pc.o_width + i * pc.o_width + j)
                      : 0.0f;
            }
          }
          // 这里使用.block()形式比for循环的速度快
          // Here using the .block() form is faster than the for loop
          dA_tmp.block(i * pc.stride, j * pc.stride, pc.height, pc.width) =
              data1;
        }
        if (pc.type == 1) {
          // 取平均值情况下的反向求导
          // Backward derivation in the case of averages
        }
      }
    }
    dA.block(0, m * size2, 1, size2) = dA_tmp.reshaped<RowMajor>(1, size2);
  }
}