# MatMul层（MatMul Layer）

## 1. 计算方法（Computation Method）

### 1.1 正向传播（Forward Propagation）
$$
Y = X \cdot W
$$

### 1.2 反向传播（Backward Propagation）
$$
\frac{\partial L}{\partial X} = W^T \cdot \frac{\partial L}{\partial Y}
$$

$$
\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}
$$
