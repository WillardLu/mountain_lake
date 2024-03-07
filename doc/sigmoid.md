# Sigmoid层（Sigmoid Layer）

## 1. 正向传播计算方法（Forward propagation calculation method）
$$
h(x) = \frac{1}{1 + e^{-x}}
$$

## 2. 反向传播计算方法（Backpropagation calculation method）

![sigmoid layer](images/sigmoid.png)

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Z} Z (1 - Z)
$$