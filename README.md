# mountain_lake
![mountain lake](cover.png)

## 1. 项目简介（Project Introduction）
本项目使用C++语言定义了一个神经网络类和实现各种功能的层。用户可以通过TOML配置文件灵活设置神经网络的结构和参数，并使用这个神经网络进行训练和预测。

This project defines a neural network class and layers implementing various functions using C++. Users can flexibly set the structure and parameters of the neural network through the TOML configuration file and use this neural network for training and prediction.

## 2. 环境（Environment）
- Ubuntu 22.04
- g++ 11.4.0
- cmake 3.22.1
- Intel i5-6500 CPU @ 3.20GHz × 4

构建所需条件请查看 CMakeLists.txt 文件。

The required conditions for building are checked in the CMakeLists.txt file.

## 3. 依赖库（Dependencies）

### 3.1 线性代数库 Eigen（Linear Algebra Library: Eigen）
Eigen 3.4.0

Ubuntu下的安装命令（Installation commands under Ubuntu）：

```bash
sudo apt install libeigen3-dev
```

### 3.2 测试工具 GoogleTest（Testing Tool: GoogleTest）
GoogleTest 1.11.0

Ubuntu下的安装命令（Installation commands under Ubuntu）：

```bash
sudo apt install libgtest-dev
```

## 4. 项目结构（Project Structure）
项目核心部分保存在 mountain_lake 文件夹中，其中包含两个文件夹，分别是：
- layers：定义各种功能的层
- neural_network：定义神经网络类

## 5. 配置文件内容与格式（Configuration File Content and Format）

### 5.1 内容构成（Content Composition）
配置文件的内容按照TOML文件中的“表”来组织，包含以下几个部分：

The content of the configuration file is organized according to the "tables" in the TOML file and contains the following sections:

#### 5.1.1 神经网络结构（Neural Network Structure）
表的名称为：neural_network。此表目前只有一个内容，对应的键名为“struct”，即神经网络结构，里面使用数组的形式定义了神经网络中各个功能层的类型、名称、输出大小等内容。

The name of the table is: neural_network, there is only one content in this table, the corresponding key name is "struct", i.e., the structure of the neural network, which uses the form of an array to define the type, name, output size, etc. of each functional layer in the neural network.

示例一（Example 1）：
```toml
struct = ["Affine:50", "Sigmoid", "Affine:10", "SoftmaxWithLoss"]
```

示例二（Example 2）：
```toml
struct = [
  "Convolution-1",
  "ReLU",
  "Pooling-1",
  "Affine:100",
  "ReLU",
  "Affine:10",
  "SoftmaxWithLoss",
]
```

从示例中可以看到，神经网络结构的内容是由字符串数组来表示的，每个字符串表示一个功能层，功能层的具体类型和参数由字符串中的内容来确定。

From the example, it can be seen that the content of the neural network structure is represented by a string array, and each string represents a functional layer. The specific type and parameters of the functional layer are determined by the content in the string.

下面对示例中的内容进行详细说明（The content of the example will be explained in detail.）：

##### 5.1.1.1 Affine:50
此定义表示一个仿射变换层，输出大小为50。

This definition represents an affine transform layer with an output size of 50.

##### 5.1.1.2 Sigmoid
此定义表示一个Sigmoid激活函数层，因为其输出大小和输入大小相同，所以不需要特别定义。

This definition denotes a Sigmoid activation function layer that does not need to be specifically defined because its output size is the same as its input size.

##### 5.1.1.3 SoftmaxWithLoss
此定义表示由Softmax和损失函数组成的层，其输出结果只会是一个浮点数值，所以不需要定义。

This definition denotes a layer consisting of Softmax and a loss function whose output will only be a floating point value, so no definition is needed.

##### 5.1.1.4 Convolution-1
此定义表示一个卷积层。程序中会把“Convolution-1”中的“Convolution”类型提取出来以确定如何处理这个层。程序中通过完整名称“Convolution-1”进一步在TOML配置文件中查找这个卷积层的详细定义，下面是这个卷积层的详细定义：

This definition represents a convolutional layer. The program extracts the "Convolution" type from "Convolution-1" to determine how to handle this layer. The program looks further into the TOML configuration file by the full name "Convolution-1" to find the detailed definition of this convolutional layer, which is shown below:
```toml
# 卷积层
[Convolution-1]
pad = 0
stride = 1
channel_num = 1
filter_num = 30
filter_height = 5
filter_width = 5
```
从以上定义中可以看出，该卷积层的参数包括：pad，填充；stride，步长；channel_num，输入通道数量；filter_num，卷积核（过滤器）数量；filter_height，卷积核高度；filter_width，卷积核宽度。虽然这里提供了输入通道的配置，但因为我使用的训练数据只有一个通道，所以程序中并不会处理这个参数，等以后使用到多通道数据时再进行处理。

As you can see from the definition above, the parameters for this convolutional layer include: pad, padding; stride, step size; channel_num, number of input channels; filter_num, number of convolutional kernels (filters); filter_height, height of convolutional kernel; and filter_width, width of convolutional kernel. Although the configuration of input channels is provided here, since the training data I used has only one channel, this parameter will not be processed in the program, and will be processed later when multi-channel data is used.

##### 5.1.1.5 ReLU
此定义表示一个线性整流层。与sigmoid层一样，因为其输出大小与输入大小相同，所以不需要特别定义。

This definition denotes a linear rectifier layer. As with the sigmoid layer, no special definition is needed because its output size is the same as its input size.

##### 5.1.1.6 Pooling-1
此定义表示一个池化层。程序从"Pooling-1"中提取"Pooling"类型，以确定如何处理此层。程序通过完全名称"Pooling-1"进一步在TOML配置文件中查找该池化层的详细定义，如下所示：

This definition denotes a pooling layer. The program extracts the "Pooling" type from "Pooling-1" to determine how to handle this layer. The program looks further into the TOML configuration file by the full name "Pooling-1" to find the detailed definition of this pooling layer, as shown below:
```toml
# 池化层
[Pooling-1]
pool_height = 2
pool_width = 2
stride = 2
filter_num = 30
type = "Max"
```
从以上定义中可以看出，该池化层的参数包括：pool_height，池化核高度；pool_width，池化核宽度；stride，步长；filter_num，卷积核（过滤器）数量；type，池化类型。这里要注意的是，卷积核数量filter_num必须与此层的上一个卷积层的卷积核数量保持一致。

As can be seen from the above definition, the parameters of this pooling layer include: pool_height, pooling kernel height; pool_width, pooling kernel width; stride, step size; filter_num, number of convolution kernels (filters); type, pooling type. It is important to note here that the number of convolution kernels filter_num must be consistent with the number of convolution kernels in the previous convolutional layer of this layer.

## 6. 补充说明（Supplementary Notes）
有关各个功能层的详细介绍请看[功能层说明](doc/layers.md)。