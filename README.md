

# CNN架构深度比较与消融分析：基于CIFAR-10的实证

### 一、摘要

本研究通过系统实现和比较三种经典CNN架构(基础CNN、VGG和ResNet)及其多种变体，在CIFAR-10数据集上进行了全面的消融实验。我们重点分析了BatchNorm、池化策略、激活函数和残差连接等关键组件对模型性能的影响。实验结果表明，不同架构组件在模型准确率、训练稳定性和计算效率方面存在显著差异，为CNN架构设计提供了实证依据。

**关键词**：卷积神经网络，消融分析，CIFAR-10，BatchNorm，残差连接, 激活函数

### 二、项目背景与目标

随着深度学习的发展，CNN 已成为计算机视觉中最核心的模型之一。然而，CNN 的结构设计复杂，其性能受多种因素影响，包括网络深度、模块组合方式以及训练策略。本项目旨在：

- 比较不同 CNN 架构（CNN vs VGG vs ResNet）在图像分类任务中的表现；
- 通过模块级消融实验，分析各结构组件对性能的实际影响；
- 综合考虑准确率、模型复杂度与训练效率，进行全面对比分析。



### 三、神经网络关键模块介绍

#### 3.1 激活函数（Activation Function）

激活函数引入非线性变换，使得神经网络能够逼近任意复杂的函数，是深度模型学习非线性特征的核心机制。

- **ReLU（Rectified Linear Unit）**  
  $$ f(x) = \max(0, x) $$
  
  - **优点**：计算简单、加速收敛、缓解梯度消失。
  - **缺点**：存在 Dying ReLU 问题，部分神经元可能永远失活。
  
- **Leaky ReLU**  
  $$ f(x) = \begin{cases} x, & x > 0 \\ \alpha x, & x \leq 0 \end{cases} $$
  
  - **优点**：为负区间提供小梯度，避免神经元死亡。

  - **缺点**：需要选择合适的 $\alpha$（通常为 0.01）。
  
- **ELU（Exponential Linear Unit）**  
  $$ f(x) = \begin{cases} x, & x > 0 \\ \alpha(\exp(x)-1), & x \leq 0 \end{cases} $$
  - **优点**：平滑非线性，有助于收敛。
  
  - **缺点**：计算复杂度较高。
  
    

#### 3.2 批归一化（Batch Normalization）

批归一化（BN）用于对网络中间层的输入进行标准化，缓解梯度消失或梯度爆炸，提升训练稳定性。

- **原理**：对于每一小批样本，BN 对每一维的输入 $x$ 做如下变换：

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta
$$
其中 $\mu$ 和 $\sigma^2$ 为该维特征在 mini-batch 上的均值和方差，$\gamma$ 和 $\beta$ 为可学习的缩放和偏移参数。

- **优点**：加速训练收敛；缓解梯度消失；对初始化不敏感；具有轻微的正则化效果。

- **缺点**：对 batch size 敏感；不适用于变长输入或 RNN 结构；推理阶段需保存均值和方差。

  

#### 3.3 池化层（Pooling Layer）

池化层用于降低特征图的空间维度，减少计算量并增强模型对位置偏移的鲁棒性。

- **最大池化（Max Pooling）**：取窗口内最大值  
  $$ y = \max(\text{window}) $$

- **平均池化（Average Pooling）**：取窗口内平均值  
  $$ y = \frac{1}{n} \sum_{i=1}^n x_i $$

- **全局池化（Global Pooling）**：将整张特征图汇聚为一个值  
  用于替代全连接层，减少参数量。

- **自适应池化（Adaptive Pooling）**：自动调整池化窗口，输出固定尺寸。

**比较**：

| 池化类型   | 优点                   | 缺点               | 典型应用             |
| ---------- | ---------------------- | ------------------ | -------------------- |
| 最大池化   | 强调显著特征，鲁棒性高 | 丢失细节           | 图像识别、目标检测   |
| 平均池化   | 保留区域整体信息       | 平滑、弱化重要特征 | 背景建模             |
| 全局池化   | 无参数，结构简洁       | 丢失空间结构       | 最终分类前的特征汇聚 |
| 自适应池化 | 灵活适配不同输入尺寸   | 实现复杂           | 输入尺寸不固定场景   |



#### 3.4 残差结构（Residual Connection）

**原理**：残差结构最早由 ResNet 提出，解决了深层网络难以训练的问题。其核心思想是通过“跳跃连接”学习残差函数：
$$
y = F(x) + x
$$
其中 $F(x)$ 为主分支卷积操作，$x$ 为输入。

**优点**：缓解梯度消失；更易训练深层模型；允许网络学习恒等映射。



#### 3.5 Dropout 正则化

Dropout 是一种防止过拟合的正则化方法，在训练过程中以一定概率随机“丢弃”神经元，从而减少模型对特定特征的依赖。

**原理**：对于隐藏层的每个神经元，以概率 $p$ 将其输出置为 0，训练阶段随机子网络组合，推理阶段使用全网络并对权重进行缩放。

**优点**：减少过拟合；模拟模型集成；提升泛化能力。

**缺点**：训练速度稍慢；与 BN 同时使用需谨慎（通常 Dropout 用于后期层）。



### 四、数据集与预处理

**数据集：** CIFAR-10，包含 10 个类别，共 60,000 张 32×32 彩色图像（50,000 用于训练，10,000 用于测试）。

**预处理流程：**

- 图像标准化处理（mean = [0.4914, 0.4822, 0.4465]，std = [0.2023, 0.1994, 0.2010]）
- 数据增强：随机水平翻转、随机裁剪（padding = 4）



### 五、模型设计与实验设置

#### 5.1 模型架构

- **CNN模型**

  支持多种激活函数、Batch Normalization、不同类型的池化操作以及可选的 Dropout

  **网络结构**

  | 层级  | 操作                   | 输出通道 | 激活 | BN   | Pooling      | 输出尺寸 (以32x32为例) |
  | ----- | ---------------------- | -------- | ---- | ---- | ------------ | ---------------------- |
  | 输入  | -                      | 3        | -    | -    | -            | 3×32×32                |
  | Conv1 | Conv2d(3→128, 3x3)     | 128      | 可选 | 可选 | 可选         | 128×16×16（使用池化）  |
  | Conv2 | Conv2d(128→256, 3x3)   | 256      | 可选 | 可选 | 可选         | 256×8×8                |
  | Conv3 | Conv2d(256→512, 3x3)   | 512      | 可选 | 可选 | 可选         | 512×4×4                |
  | Conv4 | Conv2d(512→1024, 3x3)  | 1024     | 可选 | 可选 | 可选         | 1024×2×2               |
  | GAP   | AdaptiveAvgPool2d(1×1) | 1024     | -    | -    | -            | 1024×1×1               |
  | FC1   | Linear(1024→4096)      | -        | 可选 | -    | Dropout 可选 | -                      |
  | FC2   | Linear(4096→10)        | -        | -    | -    | -            | -                      |

​	

- **VGG模型**
   基于 VGG16 模型的一个扩展版本，采用了不同的池化方法、批归一化（Batch Normalization）和激活函数等改进。

   **网络结构chazhao**

   | 层级       | 操作                            | 输出通道 | 激活函数 | BN   | Pooling | 输出尺寸  |
   | ---------- | ------------------------------- | -------- | -------- | ---- | ------- | :-------: |
   | 输入       | -                               | 3        | -        | -    | -       |  3×32×32  |
   | Conv1      | Conv2d(3→64, 3x3, padding=1)    | 64       | 可选     | 可选 | -       | 64×32×32  |
   | Conv2      | Conv2d(64→64, 3x3, padding=1)   | 64       | 可选     | 可选 | -       | 64×32×32  |
   | MaxPool1   | MaxPool2d(2x2, stride=2)        | 64       | -        | -    | 可选    | 64×16×16  |
   | Conv3      | Conv2d(64→128, 3x3, padding=1)  | 128      | 可选     | 可选 | -       | 128×16×16 |
   | Conv4      | Conv2d(128→128, 3x3, padding=1) | 128      | 可选     | 可选 | -       | 128×16×16 |
   | MaxPool2   | MaxPool2d(2x2, stride=2)        | 128      | -        | -    | 可选    |  128×8×8  |
   | Conv5      | Conv2d(128→256, 3x3, padding=1) | 256      | 可选     | 可选 | -       |  256×8×8  |
   | Conv6      | Conv2d(256→256, 3x3, padding=1) | 256      | 可选     | 可选 | -       |  256×8×8  |
   | Conv7      | Conv2d(256→256, 3x3, padding=1) | 256      | 可选     | 可选 | -       |  256×8×8  |
   | MaxPool3   | MaxPool2d(2x2, stride=2)        | 256      | -        | -    | 可选    |  256×4×4  |
   | Conv8      | Conv2d(256→512, 3x3, padding=1) | 512      | 可选     | 可选 | -       |  512×4×4  |
   | Conv9      | Conv2d(512→512, 3x3, padding=1) | 512      | 可选     | 可选 | -       |  512×4×4  |
   | Conv10     | Conv2d(512→512, 3x3, padding=1) | 512      | 可选     | 可选 | -       |  512×4×4  |
   | MaxPool4   | MaxPool2d(2x2, stride=2)        | 512      | -        | -    | 可选    |  512×2×2  |
   | Conv11     | Conv2d(512→512, 3x3, padding=1) | 512      | 可选     | 可选 | -       |  512×2×2  |
   | Conv12     | Conv2d(512→512, 3x3, padding=1) | 512      | 可选     | 可选 | -       |  512×2×2  |
   | MaxPool5   | MaxPool2d(2x2, stride=2)        | 512      | -        | -    | 可选    |  512×1×1  |
   | GlobalPool | AdaptiveAvgPool2d(1×1)          | 512      | -        | -    | -       |  512×1×1  |
   | Dropout1   | Dropout(p=dropout_rate1)        | -        | -        | -    | -       |     -     |
   | FC1        | Linear(512→4096)                | 4096     | 可选     | -    | -       |     -     |
   | Dropout2   | Dropout(p=dropout_rate2)        | -        | -        | -    | -       |     -     |
   | FC2        | Linear(4096→4096)               | 4096     | 可选     | -    | -       |     -     |
   | FC3        | Linear(4096→10)                 | 10       | -        | -    | -       |     -     |



- **ResNet模型**

  自定义的轻量级残差神经网络，基于 ResNet 思想进行设计。

  **网络结构**

  | 层级     | 操作                                      | 输出通道 | 激活 | BN   | 残差 | Pooling | 输出尺寸  |
  | :------- | :---------------------------------------- | -------- | ---- | ---- | ---- | ------- | --------- |
  | 输入     | -                                         | 3        | -    | -    | -    | -       | 3×32×32   |
  | Conv1    | Conv2d(3→64, 3×3, padding=1)              | 64       | 可选 | 可选 | -    | -       | 64×32×32  |
  | Layer1-1 | BasicBlock(64→64, stride=1)               | 64       | 可选 | 可选 | 可选 | -       | 64×32×32  |
  | Layer1-2 | BasicBlock(64→64, stride=1)               | 64       | 可选 | 可选 | 可选 | -       | 64×32×32  |
  | Pool1    | MaxPool2d(2×2)                            | 64       | -    | -    | -    | 可选    | 64×16×16  |
  | Layer2-1 | BasicBlock(64→128, stride=2, Downsample)  | 128      | 可选 | 可选 | 可选 | -       | 128×16×16 |
  | Layer2-2 | BasicBlock(128→128, stride=1)             | 128      | 可选 | 可选 | 可选 | -       | 128×16×16 |
  | Pool2    | MaxPool2d(2×2)                            | 128      | -    | -    | -    | 可选    | 128×8×8   |
  | Layer3-1 | BasicBlock(128→256, stride=2, Downsample) | 256      | 可选 | 可选 | 可选 | -       | 256×8×8   |
  | Layer3-2 | BasicBlock(256→256, stride=1)             | 256      | 可选 | 可选 | 可选 | -       | 256×8×8   |
  | Pool3    | MaxPool2d(2×2)                            | 256      | -    | -    | -    | 可选    | 256×4×4   |
  | Layer4-1 | BasicBlock(256→512, stride=2, Downsample) | 512      | 可选 | 可选 | 可选 | -       | 512×4×4   |
  | Layer4-2 | BasicBlock(512→512, stride=1)             | 512      | 可选 | 可选 | 可选 | -       | 512×4×4   |
  | GAP      | AdaptiveAvgPool2d(1×1)                    | 512      | -    | -    | -    | -       | 512×1×1   |
  | Flatten  | -                                         | -        | -    | -    | -    | -       | 512       |
  | FC       | Linear(512→10)                            | -        | -    | -    | -    | 可选    | 10        |



#### 5.2 实验配置

| 项目          | 配置                           |
| ------------- | :----------------------------- |
| Optimizer     | SGD (momentum=0.9)             |
| Learning Rate | 初始 0.01，StepLR 每 10 轮减半 |
| Batch Size    | 128                            |
| Epochs        | 120                            |

### 六、消融实验设置

我们分别在CNN、 VGG 和 ResNet模型上移除或替换以下模块进行消融实验：

- **Batch Normalization**
- **残差连接（仅对 ResNet）**
- **Dropout（p=0.5）**
- **池化方式（MaxPool → AvgPool）**
- **激活函数 (ReLU→ LeakyReLU → ELU)**



#### 6.1 **CNN模型配置**

| 配置          | BatchNorm | 池化类型 | 激活函数  | Dropout |
| ------------- | --------- | -------- | --------- | ------- |
| CNN_Baseline  | √         | Max      | ReLU      | ×       |
| CNN_Dropout   | √         | Max      | ReLU      | √       |
| CNN_No_BN     | ×         | Max      | ReLU      | ×       |
| CNN_AvgPool   | √         | Avg      | ReLU      | ×       |
| CNN_No_Pool   | √         | None     | ReLU      | ×       |
| CNN_No_BN_Avg | ×         | Avg      | ReLU      | ×       |
| CNN_LeakyReLU | √         | Max      | LeakyReLU | ×       |
| CNN_ELU       | √         | Max      | ELU       | ×       |



#### 6.2 ResNet模型配置

| 配置                 | BatchNorm | 残差连接 | 池化类型 | 激活函数  | Dropout |
| -------------------- | --------- | -------- | -------- | --------- | ------- |
| ResNet_Baseline      | √         | √        | Max      | ReLU      | ×       |
| ResNet_No_Res        | √         | ×        | Max      | ReLU      | ×       |
| ResNet_No_BN         | ×         | √        | Max      | ReLU      | ×       |
| ResNet_AvgPool       | √         | √        | Avg      | ReLU      | ×       |
| ResNet_No_BN_AvgPool | ×         | √        | Avg      | ReLU      | ×       |
| ResNet_Dropout       | √         | √        | Max      | ReLU      | √       |
| ResNet_LeakyReLU     | √         | √        | Max      | LeakyReLU | ×       |
| ResNet_ELU           | √         | √        | Max      | ELU       | ×       |
| ResNet_No_Pool       | √         | √        | None     | ReLU      | ×       |



#### 6.3 VGG模型配置

| 配置          | BatchNorm | 池化类型 | 激活函数  | Dropout |
| ------------- | --------- | -------- | --------- | ------- |
| VGG_Baseline  | √         | Max      | ReLU      | ×       |
| VGG_AvgPool   | √         | Avg      | ReLU      | ×       |
| VGG_Dropout   | √         | Max      | ReLU      | √       |
| VGG_LeakyReLU | √         | Max      | LeakyReLU | ×       |
| VGG_ELU       | √         | Max      | ELU       | ×       |



### 七、可视化结果与分析

#### 7.1 模型性能指标

| Model                   | Best Test Acc (%) | Training Time (min) | Params  | FLOPs    |
| ----------------------- | ----------------- | ------------------- | ------- | -------- |
| CNN_v2_AvgPool          | 90.61             | 65.11               | 10.442M | 235.316M |
| CNN_v2_Baseline         | 91.27             | 50.92               | 10.442M | 235.255M |
| CNN_v2_Dropout          | 90.63             | 54.18               | 10.442M | 235.255M |
| CNN_v2_ELU              | 89.57             | 53.02               | 10.442M | 235.255M |
| CNN_v2_LeakyReLU        | 91.36             | 53.8                | 10.442M | 235.255M |
| CNN_v2_NoPooling        | 88.27             | 382.54              | 10.442M | 6.358G   |
| CNN_v2_No_BN_Avg        | 86.09             | 64.71               | 10.438M | 234.333M |
| CNN_v2_No_BN            | 89.06             | 52.92               | 10.438M | 234.272M |
| ResNet_v2_AvgPool       | 92.51             | 71.94               | 11.172M | 204.587M |
| ResNet_v2_Baseline      | 91.75             | 76.55               | 11.172M | 204.569M |
| ResNet_v2_Dropout       | 91.73             | 66.83               | 11.172M | 204.569M |
| ResNet_v2_ELU           | 89.96             | 71.43               | 11.172M | 204.569M |
| ResNet_v2_LeakyReLU     | 91.8              | 71.89               | 11.172M | 204.569M |
| ResNet_v2_No_BN_AvgPool | 89.97             | 68.94               | 11.169M | 203.121M |
| ResNet_v2_No_BN         | 90.24             | 69.23               | 11.169M | 203.102M |
| ResNet_v2_No_Pool       | 94.12             | 99.71               | 11.172M | 557.660M |
| ResNet_v2_No_Res        | 91.63             | 83.13               | 11.000M | 203.782M |
| VGG_v2_AvgPool          | 92.82             | 100.02              | 33.647M | 333.250M |
| VGG_v2_Baseline         | 92.67             | 78.76               | 33.647M | 333.219M |
| VGG_v2_Dropout          | 92.68             | 78.7                | 33.647M | 333.219M |
| VGG_v2_ELU              | 91.32             | 78.16               | 33.647M | 333.219M |
| VGG_v2_LeakyReLU        | 92.88             | 78.43               | 33.647M | 333.219M |



##### 7.1.1 **最佳测试准确率**（Best Test Accuracy）

- **VGG_v2系列**的表现相对优异，尤其是**VGG_v2_LeakyReLU**，其最佳测试准确率为**92.88%**。其他VGG_v2模型的表现也相近，准确率都在92%左右。该系列模型平均准确率为**92.51%**。
- **ResNet_v2系列**的表现也非常出色，**ResNet_v2_AvgPool**的最佳测试准确率为**92.51%**，略低于VGG_v2_LeakyReLU。其他ResNet_v2模型的准确率大多集中在91%~92%之间，其中**ResNet_v2_No_Pool**的表现特别突出，达到了**94.12%**，这是所有模型中准确率最高的。该系列模型平均准确率为**91.52%**。
- **CNN_v2系列**的准确率普遍低于VGG_v2和ResNet_v2。最好的表现来自**CNN_v2_LeakyReLU**，其准确率为**91.36%**，而**CNN_v2_No_BN_Avg**则表现最差，准确率为**86.09%**。该系列模型平均准确率为**89.61%**。



![acc_comparison_top3](D:\AAAAA\works-20250510T134512Z-1-001\works\figures1\acc_comparison_top3.png)



##### 7.1.2 **训练时间**（Training Time)

- **VGG_v2系列**的训练时间相对较长，特别是**VGG_v2_AvgPool**，需要**100.02分钟**，而其他VGG_v2模型的训练时间也在**78分钟**左右。
- **ResNet_v2系列**的训练时间比VGG_v2系列稍短，最短的训练时间为**66.83分钟**（**ResNet_v2_Dropout**），而**ResNet_v2_No_Pool**则需要**99.71分钟**，与**VGG_v2_AvgPool**相当。
- **CNN_v2系列**的训练时间相对较短，尤其是**CNN_v2_Baseline**，仅需要**50.92分钟**，而**CNN_v2_NoPooling**的训练时间则特别长，达到**382.54分钟**，远高于其他模型。



![train_time_comparison](D:\AAAAA\works-20250510T134512Z-1-001\works\figures1\train_time_comparison.png)



##### 7.1.3 **参数数量**（Params）

- 所有模型的参数数量基本相似，主要集中在**10M**到**34M**之间，说明这些模型的结构并没有大幅度变化。**VGG_v2系列**的参数数量相对较多，约为**33.647M**，而其他模型的参数数量大多在**10M到11M**之间。



![params_comparison](D:\AAAAA\works-20250510T134512Z-1-001\works\figures2\params_comparison.png)



##### 7.1.4 每秒浮点运算量（FLOPs）

- **VGG_v2系列**的FLOPs较高，约为**333.2M**，这也与其较大的参数量相对应。
- **ResNet_v2系列**的FLOPs相对较低，尤其是**ResNet_v2_Baseline**和其他变种（不包括ResNet_v2_No_Pool）的FLOPs都在**204.5M**左右，这使得它们在计算效率上具有一定的优势。
- **CNN_v2系列**的FLOPs在不同模型间差异较大。**CNN_v2_NoPooling**的FLOPs达到了**6.358G**，明显高于其他模型，这可能是由于其结构调整导致的计算需求增加。而其他模型的FLOPs较为接近，约为**235.2M**。



![flops_comparison](D:\AAAAA\works-20250510T134512Z-1-001\works\figures2\flops_comparison.png)



##### 7.1.5 **训练时间与性能的权衡**

- **ResNet_v2系列**提供了较为平衡的选择。**ResNet_v2_Baseline**的训练时间为**76.55分钟**，其最佳测试准确率为**91.75%**，而**ResNet_v2_No_Pool**虽然具有最高的准确率（**94.12%**），但训练时间较长，达到**99.71分钟**，表现出计算复杂度的增加。
- **VGG_v2系列**尽管有较高的准确率，尤其是**VGG_v2_AvgPool**，但由于较长的训练时间，它在实际应用中可能不如其他较为高效的模型（**ResNet_v2_Baseline**）具有优势。
- **CNN_v2系列**在训练时间上的优势明显，但由于性能较低，尤其是在**CNN_v2_NoPooling**模型中表现尤为明显，准确率相对较差。



![training_time_vs_acc](D:\AAAAA\works-20250510T134512Z-1-001\works\figures\metrics\training_time_vs_acc.png)



#### 7.2 模块影响分析

##### 7.2.1 BatchNorm移除的影响分析

| Model                   | Acc (%) | Time (min) | Params  | FLOPs    | 变化说明  |
| ----------------------- | ------- | ---------- | ------- | -------- | --------- |
| CNN_v2_Baseline         | 91.27   | 50.92      | 10.442M | 235.255M | 有BN      |
| CNN_v2_No_BN            | 89.06   | 52.92      | 10.438M | 234.272M | Acc ↓2.21 |
| CNN_v2_No_BN_Avg        | 86.09   | 64.71      | 10.438M | 234.333M | Acc ↓5.18 |
| ResNet_v2_Baseline      | 91.75   | 76.55      | 11.172M | 204.569M | 有BN      |
| ResNet_v2_No_BN         | 90.24   | 69.23      | 11.169M | 203.102M | Acc ↓1.51 |
| ResNet_v2_No_BN_AvgPool | 89.97   | 68.94      | 11.169M | 203.121M | Acc ↓1.78 |

![bn](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\bn.png)



- **准确率（Acc）**
  BatchNorm的移除对模型准确率产生了**显著的负面影响**。在CNN_v2架构中，移除BN导致准确率**下降2.21%**（91.27%→89.06%），若进一步采用平均池化替代，下降幅度扩大到**5.18%**。ResNet_v2架构表现相对稳健，但仍有**1.51%-1.78%**的精度损失。



![Best Test Acc (%)_horizontal_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\BatchNorm Removal\Best Test Acc (%)_horizontal_bar.png)



- **训练时间（Time）**
  **CNN_v2**训练时长**增加约4%**（50.92→52.92分钟），特别是**CNN_v2_No_BN_Avg模型**的时间成本**激增27%**（50.92→64.71分钟），说明平均池化与无BN的叠加会**显著降低收敛速度**。但是，**ResNet_v2**在移除BN后，训练时间**减少约9.5%**（76.55→69.23分钟），与CNN_v2的表现相反。



![Training Time (min)_dot_plot](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\BatchNorm Removal\Training Time (min)_dot_plot.png)



- **计算量（FLOPs）**

  **BN对FLOPs的影响可忽略不计**。CNN_v2减少**0.42%**，ResNet_v2减少**0.72%**，即使结合平均池化（CNN_v2_No_BN_Avg），FLOPs仍仅减少**0.39%**（234.333M vs 235.255M)。
  
  

![FLOPs_vertical_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\BatchNorm Removal\FLOPs_vertical_bar.png)



**BatchNorm (BN)** **对模型的稳定性和准确率提升尤为重要**。对于基础CNN架构，移除BN后准确率从91.27%下降到89.06%，降幅达2.21%；若同时使用平均池化，准确率甚至降幅达到5.18%。而ResNet架构中，移除BN后的准确率下降较小，仅为1.51%（91.75%→90.24%）到1.78%。这表明BN在浅层网络中对保持特征分布稳定至关重要，它通过正则化输入特征使得梯度传播更高效；缺少BN后，浅层CNN的特征提取能力受到明显影响。另一方面，**BN的去除对模型训练时间的影响因网络结构而异**：在CNN中，缺失BN会导致训练时间增加约4%，而在同时采用平均池化时，时间更是增长了27%。这说明BN本身具有隐式正则化和加速收敛的作用，一旦失去，网络需要更多迭代来收敛。**值得注意的是**，对于ResNet，去除BN后训练时间反而缩短了约9.5%。这可能是因为ResNet的残差结构本身就有助于梯度流动，部分替代了BN的稳定功能；与此同时，移除BN减少了额外的计算开销，使得训练效率提高。无论在CNN还是ResNet中，BN的去除对参数量和FLOPs几乎没有影响（变化均在1%以内），说明BN主要影响的是训练动态而不是网络规模。

 

##### 7.2.2 Dropout加入的影响分析

| Model              | Acc (%) | Time (min) | Params  | FLOPs    | 变化说明         |
| ------------------ | ------- | ---------- | ------- | -------- | ---------------- |
| CNN_v2_Baseline    | 91.27   | 50.92      | 10.442M | 235.255M | 无 Dropout       |
| CNN_v2_Dropout     | 90.63   | 54.18      | 10.442M | 235.255M | Acc ↓0.64，时间↑ |
| ResNet_v2_Baseline | 91.75   | 76.55      | 11.172M | 204.569M | 无 Dropout       |
| ResNet_v2_Dropout  | 91.73   | 66.83      | 11.172M | 204.569M | Acc ↓0.08        |
| VGG_v2_Baseline    | 92.67   | 78.76      | 33.647M | 333.219M | 无 Dropout       |
| VGG_v2_Dropout     | 92.68   | 78.70      | 33.647M | 333.219M | 变化几乎无       |

![do](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\do.png)



- **准确率（Acc）**

  **CNN_v2**的Dropout导致**轻微准确率下降**（91.27%→90.63%），**ResNet_v2和VGG_v2**的Dropout**几乎不影响**准确率。



![Best Test Acc (%)_horizontal_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Dropout Addition\Best Test Acc (%)_horizontal_bar.png)



- **训练时间（Time）**

  **CNN_v2**训练时间**增加6.4%**（50.92→54.18分钟）。**ResNet_v2**训练时间**减少12.7%**（76.55→66.83分钟），对**VGG_v2**训练时间**无显著影响**。



![image-20250515164343260](C:\Users\mo\AppData\Roaming\Typora\typora-user-images\image-20250515164343260.png)



- **计算量（FLOPs）**

  Dropout**不影响FLOPs**

![FLOPs_horizontal_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2/Dropout Addition\FLOPs_horizontal_bar.png)



**Dropout** **模块的影响相对温和**。在引入Dropout后，**CNN_v2** 的准确率略有下降（从91.27%降至90.63%，降低0.64%），训练时间增加6.4%。这表明对于已接近性能上限的浅层CNN而言，Dropout并没有带来准确率提升，反而降低了部分性能并需要更长时间收敛。而在**ResNet_v2** 和 **VGG_v2**中，Dropout几乎不改变准确率（ResNet下降仅0.08%，VGG上升0.01%），因为它们的网络结构已经具有较强的正则特性：ResNet的残差连接和VGG的稠密卷积已经在一定程度上发挥了正则化作用。因此，在这些深层模型中加入Dropout效果不明显。此外，ResNet加入Dropout后训练时间意外地缩短了约12.7%（76.55→66.83分钟），可能是实验波动或Dropout的随机梯度扰动帮助网络更快跳出鞍点；VGG的训练时间则几乎保持不变，表明其计算瓶颈主要在卷积层，Dropout对时间影响不大。Dropout在训练时引入随机丢弃操作，**不影响模型的推理FLOPs**，因此对计算量没有太大变化。



##### 7.2.3 激活函数替换的影响

| Model  | Variant   | Acc (%) | Time (min) | Params  | FLOPs    | 总结                |
| ------ | --------- | ------- | ---------- | ------- | -------- | ------------------- |
| CNN_v2 | Baseline  | 91.27   | 50.92      | 10.442M | 235.255M | ReLU                |
|        | LeakyReLU | 91.36   | 53.8       | 10.442M | 235.255M | Acc ↑0.09，时间略增 |
|        | ELU       | 89.57   | 53.02      | 10.442M | 235.255M | Acc ↓1.7            |

![CNN](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\CNN.png)



| Model     | Variant   | Acc (%) | Time (min) | Params  | FLOPs    | 总结                     |
| --------- | --------- | ------- | ---------- | ------- | -------- | ------------------------ |
| ResNet_v2 | Baseline  | 91.75   | 76.55      | 11.172M | 204.569M | ReLU                     |
|           | LeakyReLU | 91.8    | 71.89      | 11.172M | 204.569M | Acc 几乎相同，复杂度大增 |
|           | ELU       | 89.96   | 71.43      | 11.172M | 204.569M | Acc ↓1.85，训练慢        |

![ResNet](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\ResNet.png)



| Model  | Variant   | Acc (%) | Time (min) | Params  | FLOPs    | 总结      |
| ------ | --------- | ------- | ---------- | ------- | -------- | --------- |
| VGG_v2 | Baseline  | 92.67   | 78.76      | 33.647M | 333.219M | ReLU      |
|        | LeakyReLU | 92.88   | 78.43      | 33.647M | 333.219M | Acc ↑0.21 |
|        | ELU       | 91.32   | 78.16      | 33.647M | 333.219M | Acc ↓1.35 |

![VGG](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\VGG.png)



- **准确率（Acc）**
  **LeakyReLU**在**VGG_v2**表现**最佳**（+0.21%），而**ELU**在**ResNet_v2**中表现**最差**（-1.85%）。



![Best Test Acc (%)_dot_plot](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Activation Function Comparison\Best Test Acc (%)_dot_plot.png)



- **训练时间（Time）**
  **LeakyReLU**使**ResNet_v2**训练时间**增加11.6%**（64.42→71.89分钟）。**ELU**由于涉及指数运算，在**CNN_v2**中**增加4.1%**时间成本（50.92→53.02分钟），**ResNet_v2**中**增加10.8%**的时间成本（64.2→71.43分钟）。**VGG_v2**的时间**变化微乎其微**。



![Training Time (min)_vertical_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Activation Function Comparison\Training Time (min)_vertical_bar.png)



- **计算量（FLOPs）**
  所有变体的FLOPs严格保持一致，因为**激活函数替换不改变网络拓扑**。



![FLOPs_horizontal_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Activation Function Comparison\FLOPs_horizontal_bar.png)



将**ReLU**替换为**LeakyReLU**后，**VGG_v2** 的准确率最高提升了0.21%（92.67→92.88%），而CNN_v2略微增0.09%，ResNet_v2几乎保持不变。这说明LeakyReLU对抑制神经元死亡、在浅层网络中保留小幅负区间梯度是有益的；但对于深层ResNet，由于已存在的稀疏激活特性和残差稳定效果，LeakyReLU带来的边际效益有限。相比之下，使用**ELU**时多数模型准确率都有明显下降：在ResNet_v2上下降1.85%（91.75→89.96%），CNN_v2上下降1.70%，VGG_v2下降1.35%。ELU 的指数运算虽能平滑负区间，但在深层网络中容易引起梯度弥散，在浅层网络中数值运算也更昂贵，不利于收敛。对应地，LeakyReLU 和 ELU 都会增加训练时间：LeakyReLU 需要保持负区间的小梯度，ResNet_v2的训练时长相对增长了数分钟；ELU由于指数运算，CNN_v2训练时长增加约4%，ResNet_v2增加约10%。VGG_v2的训练时间几乎未变，进一步证明当计算主要花费在卷积计算时，激活函数复杂度带来的时间开销可忽略不计。所有激活替换实验中，模型的FLOPs完全相同，因为网络拓扑并未发生变化。



##### 7.2.4 残差连接移除分析

| Model              | Acc (%) | Time (min) | Params  | FLOPs    | 对比说明                         |
| ------------------ | ------- | ---------- | ------- | -------- | -------------------------------- |
| ResNet_v2_Baseline | 91.75   | 76.55      | 11.172M | 204.569M | 有残差连接                       |
| ResNet_v2_No_Res   | 91.63   | 83.13      | 11.000M | 203.782M | Acc ↓0.12，Params，FLOPs略有下降 |

![res](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\res.png)



- **准确率（Acc）**
  ResNet_v2_No_Res仅**下降0.12%**（91.75%→91.63%）。



![Best Test Acc (%)_dot_plot](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Residual Connection Removal\Best Test Acc (%)_dot_plot.png)



- **训练时间（Time）**

  训练时间**增长8.6%** 。

![Training Time (min)_horizontal_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Residual Connection Removal\Training Time (min)_horizontal_bar.png)



- **计算量（FLOPs）**
  参数量和FLOPs**减少**。



![FLOPs_vertical_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Residual Connection Removal\FLOPs_vertical_bar.png)



去除残差后，模型准确率仅从91.75%略降至91.63%（下降0.12%），远低于理论上对超深网络造成的影响。这主要是因为实验中使用的ResNet仅有18层，网络深度相对较浅，残差连接的优势尚不明显。但移除残差却明显拉长了训练时间，增长了约8.6%。这表明**残差连接的恒等映射路径有效加速了梯度传播，减少收敛所需的迭代次数**。去除残差略微减少了参数量和FLOPs（约0.3%以内），因为少了分支合并的计算，但这种减少对性能影响可忽略。



##### 7.2.5 池化方式变化研究

| Model            | Acc (%) | Time (min) | Params  | FLOPs    | 说明                            |
| ---------------- | ------- | ---------- | ------- | -------- | ------------------------------- |
| CNN_v2_Baseline  | 91.27   | 50.92      | 10.442M | 235.255M | MaxPool                         |
| CNN_v2_AvgPool   | 90.61   | 65.11      | 10.442M | 235.316M | AvgPool，Acc ↓0.66，时间↑       |
| CNN_v2_NoPooling | 88.27   | 382.54     | 10.442M | 6.358G   | 无Pooling，Acc ↓3%，计算量↑26倍 |

![cnnpool](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\cnnpool.png)



| Model              | Acc (%) | Time (min) | Params  | FLOPs    | 说明                           |
| ------------------ | ------- | ---------- | ------- | -------- | ------------------------------ |
| ResNet_v2_Baseline | 91.75   | 76.55      | 11.172M | 204.569M | MaxPool                        |
| ResNet_v2_AvgPool  | 92.51   | 71.94      | 11.172M | 204.587M | AvgPool 反而表现更优           |
| ResNet_v2_No_Pool  | 94.12   | 99.71      | 11.172M | 557.660M | 无Pool，但训练慢、计算成本极高 |

![resnetpool](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\resnetpool.png)



- **准确率（Acc）**
  最大池化在CNN_v2中最优表现91.27%，平均池化降低0.66%，无池化暴跌3%。



![Best Test Acc (%)_horizontal_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Pooling Strategy Comparison\Best Test Acc (%)_horizontal_bar.png)



- **训练时间（Time）**
  无池化策略倒是训练时间**大幅增加**：CNN_v2_NoPooling训练时间激增651%（50.92→382.54分钟），ResNet_v2_No_Pool时间增加30%。



![Training Time (min)_vertical_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Pooling Strategy Comparison\Training Time (min)_vertical_bar.png)



- **计算量（FLOPs）**
  **CNN_v2**无池化导致FLOPs**增长26倍**（235M→6.4G）。**ResNet_v2**无池化仅**增长2.7倍**（204.569M→558M）。



![FLOPs_vertical_bar](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\Pooling Strategy Comparison\FLOPs_vertical_bar.png)

在**池化策略**的对比中，发现**不同架构对池化方式的依赖程度大不相同**。在**CNN_v2**模型中，默认的最大池化效果最佳（91.27%），使用平均池化准确率略降0.66%，完全取消池化（即不下采样）则使准确率暴跌至88.27%（下降3%），同时训练时间激增651%，FLOPs激增26倍。这说明浅层CNN缺乏其他下采样机制时，必须依赖池化层来控制特征图尺寸和计算量，否则特征图尺寸迅速膨胀使得计算变得难以承受。相比之下，**ResNet_v2**通过在卷积层中使用步幅为2的卷积实现下采样，其隐式池化能力更强：换用平均池化后准确率反而略升至92.51%，而完全去除池化时准确率达到全局最高94.12%。这表明在残差结构中可以利用学得的卷积核自行完成信息压缩，从而提升精度。然而，ResNet_v2无池化版本的训练时间仍增加了30%，FLOPs增长2.7倍，说明即使其瓶颈结构控制了一定计算量，特征图尺寸扩大也对计算资源有较大影响。**综上，池化策略与网络结构相互作用显著：在浅层CNN中必须使用适当的池化方式来保持效率，而在深层ResNet中可以灵活选择以换取更高精度。**



#### 7.3 总结

**BatchNorm**是最关键的正则化组件，它能显著提升训练的稳定性和模型精度。在缺失BN的情况下，浅层CNN的性能损失最为明显，训练时间也显著增加，因此在设计中应尽量保留BN层。

**Dropout**的效果相对局限，其在深层网络中作用甚微，在浅层CNN中仅略微降低了性能并延长了收敛时间。换言之，当模型本身具有较强正则能力（如ResNet的残差结构或VGG的深层结构）时，额外的Dropout带来的收益很有限。

**激活函数**的选择，我们发现LeakyReLU在所有模型中表现均不弱于标准ReLU，尤其在VGG网络中带来了微小提升；而ELU往往导致深层模型收敛变慢和精度下降，因此一般建议在常规场景下使用ReLU或LeakyReLU作为默认激活函数。

**池化策略**的选取需结合网络结构：对于CNN而言，最大池化最为高效，平均池化稍有精度损失，无池化几乎不可行；而ResNet可以利用卷积自适应下采样，平均池化甚至可略微提升精度，虽然无池化会提升计算负担，但也可换来更高的准确率。

**残差连接**大幅加速了训练收敛。对于更深层的网络，残差连接的重要性将更加凸显。

各架构的**综合表现**对比如下：

| 架构类型      | 最佳变体          | 最佳准确率 (%) | 平均准确率 (%) | 参数量 | FLOPs 范围         | 平均训练时间 |
| ------------- | ----------------- | -------------- | -------------- | ------ | ------------------ | ------------ |
| **ResNet_v2** | ResNet_v2_No_Pool | **94.12**      | 91.52          | ≈11.2M | 204M – 558M        | ≈72 分钟     |
| **VGG_v2**    | VGG_v2_LeakyReLU  | **92.88**      | 92.51          | ≈33.6M | ≈333M              | ≈79 分钟     |
| **CNN_v2**    | CNN_v2_LeakyReLU  | **91.36**      | 89.61          | ≈10.4M | ≈235M（极端 6.4G） | ≈51 分钟     |

VGG_v2 系列在准确率上略胜一筹，最佳模型达到92.88%，但其参数规模（≈33.6M）和FLOPs（≈333M）也最大，训练时间最长（约79分钟）；ResNet_v2 系列平衡度较好，平均准确率91.52%，最佳模型（无池化变体）达94.12%，参数量适中（≈11.2M），计算量在204–558M之间，平均训练时间约72分钟；基础CNN_v2 系列准确率最低（平均89.61%），但其训练速度最快（基线模型约51分钟），参数量最小（≈10.4M）。



### 八、多组件融合训练的模型探索

在第七章的基础上，进一步研究了多种组件同时作用时的网络行为。具体而言，选取了BatchNorm、Dropout和激活函数几种核心模块，构造了不同的**复合模型配置**。下表列出了所设计的主要模型组合及其在CIFAR-10上的表现指标。

#### 8.1 模型配置与性能对比

| 模型配置                        | BN   | Dropout | 池化 | 激活函数  | 测试准确率 (%) | 训练时长 (min) | 参数量  |
| ------------------------------- | ---- | ------- | ---- | --------- | -------------- | -------------- | ------- |
| CNN_v2_Baseline                 | √    | ×       | max  | ReLU      | 91.27          | 50.92          | 10.442M |
| CNN_v2_Leaky_AvgPool_Dropout    | √    | √       | avg  | LeakyReLU | 90.48          | 66.33          | 10.442M |
| CNN_v2_NoBN_ELU_Dropout         | ×    | √       | max  | ELU       | 89.21          | 66.85          | 10.438M |
| ResNet_v2_Baseline              | √    | ×       | max  | ReLU      | 91.75          | 76.55          | 11.172M |
| ResNet_v2_Leaky_AvgPool_Dropout | √    | √       | avg  | LeakyReLU | 91.79          | 76.64          | 11.172M |
| ResNet_v2_NoBN_ELU_Dropout      | ×    | √       | max  | ELU       | 90.45          | 70.66          | 11.169M |
| VGG_v2_Baseline                 | √    | ×       | max  | ReLU      | 92.67          | 78.76          | 33.647M |
| VGG_v2_Leaky_AvgPool_Dropout    | √    | √       | avg  | LeakyReLU | 92.97          | 86.72          | 33.647M |
| VGG_v2_ELU_Dropout              | ×    | √       | max  | ELU       | 90.84          | 86.15          | 33.647M |



从表中可见，尽管融合多个模块的配置在个别情况下略有提升，但整体表现**并未显著优于最佳单模块模型**。



#### 8.2 架构分组表现分析

对于 **CNN_v2** 架构，CNN_v2_Leaky_AvgPool_Dropout 配置在保留BN的前提下引入Dropout和LeakyReLU，测试准确率为90.48%，相比基线模型（91.27%）略有下降，但训练时间显著延长。进一步地，CNN_v2_NoBN_ELU_Dropout 在去除BN的同时采用ELU和Dropout，准确率仅为89.21%，验证了**BN缺失对浅层网络影响严重**。此外，ELU与Dropout的组合加重了训练不稳定性，导致模型在有限轮数内难以收敛至理想精度。

![cnnf](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\cnnf.png)



在 **ResNet_v2** 架构中，多组件融合表现相对稳定。ResNet_v2_Leaky_AvgPool_Dropout 模型的准确率达到91.79%，基本与基线持平（91.75%），显示出ResNet的鲁棒性。相比之下，ResNet_v2_NoBN_ELU_Dropout 准确率下降至90.45%，进一步验证BN对于深层网络依然重要，而ELU在ResNet中存在训练效率和梯度传播上的隐患。

![resnetf](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\resnetf.png)



在**VGG_v2** 表现最为突出的是 VGG_v2_Leaky_AvgPool_Dropout，其融合LeakyReLU、平均池化及双Dropout，测试准确率达**92.97%**，为本轮实验中所有融合模型之最。表明VGG结构在模块融合下依然具备提升空间。相对而言，VGG_v2_ELU_Dropout 尽管结构复杂，准确率也仅为90.84%，进一步说明 ELU 并不适用于所有架构，尤其在计算复杂度较高的网络中并无优势。

![vggf](D:\AAAAA\works-20250510T134512Z-1-001\works\results_v2\train_logs\vggf.png)



#### 8.3 模块组合效应总结

实验结果显示，多组件融合并未如预期产生“叠加收益”，反而因模块间干扰引发性能波动。总结如下：

- **BatchNorm 的存在是性能保障的关键因素**，移除后性能整体下降明显；
- **Dropout 的作用随模型深度和结构正则化程度减弱**，在 VGG 和 ResNet 中增益有限；
- **LeakyReLU 是相对安全的激活替代选项**，能在多数场景中带来小幅提升；
- **ELU 激活不稳定，尤其在去除 BN 后对训练动态影响显著**；
- **池化方式的更换需结合模型架构灵活调整**，平均池化在 ResNet 和 VGG 中更易融合。



### 九、总结

本次实验围绕基础CNN、VGG和ResNet三种典型卷积神经网络架构，基于CIFAR-10图像分类任务，系统性地开展了多个维度的模型对比与模块消融实验。研究聚焦于影响模型性能的核心结构组件，包括：Batch Normalization、激活函数（ReLU、LeakyReLU、ELU）、Dropout、池化策略及残差连接等。通过构建大量可控变体并分析其在准确率、训练效率、参数量与计算复杂度等方面的表现，我们得出如下关键结论：

1. **架构对性能的基线差异显著**：
   - **VGG_v2** 系列准确率最高（平均92.51%，峰值92.88%），但代价是参数量最大、计算开销高；
   - **ResNet_v2** 在性能、效率之间取得较好平衡，具备最强鲁棒性，最高准确率达94.12%（无池化变体）；
   - **CNN_v2** 虽准确率偏低（平均89.61%），但结构简单、收敛速度快，适合轻量化部署。
2. **BatchNorm 是最关键的性能增强模块**：
    无论是在浅层CNN还是深层ResNet中，移除BatchNorm均会造成准确率下降（最高可达5%），训练时间也显著延长。BN既提供了稳定性，又兼具轻量正则化功能。
3. **激活函数对性能的影响因架构而异**：
    LeakyReLU在各网络中普遍带来小幅提升（尤其在VGG中提升明显），而ELU激活在多数情况下反而导致训练变慢、准确率下降，表明其适用性较窄。
4. **Dropout对深层模型影响有限**：
    对于已包含残差结构（ResNet）或深层堆叠（VGG）的模型，Dropout并未带来实质性性能提升，甚至可能影响收敛速度。在浅层CNN中，其正则化效应也较温和。
5. **池化策略需结构匹配**：
    CNN更适合最大池化，取消池化将严重影响其训练效率与准确率；而ResNet结构能较好吸收平均池化，甚至在无池化条件下达成最高准确率，但代价是显著的训练时间与计算资源消耗。
6. **残差连接增强了深层网络的收敛速度**：
    尽管18层ResNet在移除残差后准确率下降不大（仅0.12%），但训练时间显著增加（约+8.6%），说明残差连接对于加速训练、改善梯度流动具有重要作用。
7. **多组件融合并未带来预期的协同增益**：
    将BN、Dropout、激活函数等组件联合使用时，模型性能并未显著超越单模块最优配置。模块间存在耦合关系，盲目叠加可能导致训练波动或效果减弱，需基于架构特性设计合理的组合方案。

