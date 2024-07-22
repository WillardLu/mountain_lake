// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "convolution.h"

/// @brief 卷积层正向传播（Forward propagation of convolutional layers）
/// @param X 输入（input）
/// @param W 权重（weights）
/// @param B 偏置（bias）
/// @param O 输出（output）
/// @param cc 配置内容（Configuration contents）
void Convolution::Forward(MatrixXf &X, MatrixXf &W, MatrixXf &B, MatrixXf &O,
                          ConvConig &cc) {
  // 1. 把图片数据转化为矩阵
  // 1. Convert image data to matrix
  // 如果使用.reshaped()函数来转换的话，X的数据会以列优先的形式转换为矩阵，即使X与X_2d在
  // 定义时都使用了行优先也没有用。
  // If you use the .reshaped() function to convert, the data for X will be
  // converted to a matrix in column-first form, even though X and X_2d were
  // both defined using row-first in the are defined with row-first will not
  // help.
  MatrixXf X_2d = X.reshaped<RowMajor>(cc.i_height, cc.i_width);

  // 2. 实现填充功能
  // 2. Implementation of the fill function
  MatrixXf X_2dp;
  if (cc.pad > 0) {
    X_2dp = MatrixXf(X_2d.rows() + 2 * cc.pad, X_2d.cols() + 2 * cc.pad);
    X_2dp.block(cc.pad, cc.pad, X_2d.rows(), X_2d.cols()) = X_2d;
  } else {
    X_2dp = X_2d;
  }
  int size1 = cc.height * cc.width;
  int size2 = cc.o_height * cc.o_width;
  int site1 = 0;
  int site2 = 0;
  MatrixXf X_tmp = MatrixXf(size2, size1);
  MatrixXf W_tmp = MatrixXf(size1, 1);
  MatrixXf O_tmp = MatrixXf(size2, 1);
  MatrixXf B_tmp = MatrixXf(size2, 1);
  for (int m = 0; m < cc.number; ++m) {
    // 这里for循环的速度比.block()形式的要快
    // Here the for loop is faster than in .block() form
    for (int i = 0; i < size1; ++i) {
      W_tmp(i, 0) = W(0, m * size1 + i);
    }
    for (int i = 0; i < cc.o_height; ++i) {
      site1 = i * cc.stride;
      site2 = i * cc.o_width;
      for (int j = 0; j < cc.o_width; ++j) {
        for (int k = 0; k < cc.height; ++k) {
          for (int l = 0; l < cc.width; ++l) {
            X_tmp(site2 + j, k * cc.width + l) =
                X_2dp(site1 + k, j * cc.stride + l);
          }
        }
      }
    }
    O_tmp.noalias() = X_tmp * W_tmp;
    for (int i = 0; i < size2; ++i) {
      B_tmp(i, 0) = B(0, m);
    }
    for (int i = 0; i < size2; ++i) {
      O(0, i + m * size2) = O_tmp(i, 0) + B_tmp(i, 0);
    }
  }
}

/// @brief 卷积层反向传播（Convolutional Layer Backpropagation）
/// @param X 输入（input）
/// @param dB 卷积层偏置的导数（Derivative of the convolutional layer bias）
/// @param dO 输出参数的导数（Derivatives of output parameters）
/// @param dW 权重的导数（Derivative of the weights）
/// @param cc 配置内容（Configuration contents）
void Convolution::Backward(MatrixXf &X, MatrixXf &dB, MatrixXf &dO,
                           MatrixXf &dW, ConvConig &cc, int layer_num) {
  int size1 = cc.height * cc.width;
  int size2 = cc.o_height * cc.o_width;
  MatrixXf dO_tmp = dO.reshaped<RowMajor>(cc.number, size2);
  for (int i = 0; i < cc.number; ++i) {
    dB(0, i) = dO_tmp.row(i).sum();
  }
  // 1. 把输入数据转化为矩阵
  // 1. Converting input data into matrices
  MatrixXf X_2d = X.reshaped<RowMajor>(cc.i_height, cc.i_width);
  // 2. 实现填充功能
  // 2. Implementation of the fill function
  MatrixXf X_2dp;
  if (cc.pad > 0) {
    X_2dp = MatrixXf(X_2d.rows() + 2 * cc.pad, X_2d.cols() + 2 * cc.pad);
    X_2dp.block(cc.pad, cc.pad, X_2d.rows(), X_2d.cols()) = X_2d;
  } else {
    X_2dp = X_2d;
  }
  MatrixXf dX_tmp1 = MatrixXf::Zero(size2, 1);
  for (int m = 0; m < cc.number; ++m) {
    for (int i = 0; i < cc.height; ++i) {
      for (int j = 0; j < cc.width; ++j) {
        for (int k = 0; k < cc.o_height; ++k) {
          for (int l = 0; l < cc.o_width; ++l) {
            dX_tmp1(k * cc.o_width + l, 0) =
                X_2dp(k * cc.stride + i, l * cc.stride + j);
          }
        }
        // 这里使用.block()形式比for循环的速度快
        // Here using the .block() form is faster than the for loop
        dW(0, m * size1 + i * cc.width + j) =
            (dO_tmp.block(m, 0, 1, size2) * dX_tmp1)(0, 0);
      }
    }
  }
  // 如果这个层被放在神经网络中的第一层，则不需要计算输入信号的导数。
  // If this layer is placed in the first layer in the neural network, there is
  // no need to calculate the derivative of the input signal.
  if (layer_num >= 2) return;
}