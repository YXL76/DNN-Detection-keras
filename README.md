# DNN-Detection-keras

Forked from [PHY-AI/DNN-Detection-keras](https://github.com/PHY-AI/DNN-Detection-keras)

## 实验目的

探究`DNN`在`OFMD`通信系统中进行信道估计和信号检测的能力

## 实验内容

将`DNN`训练模型的结构与传统算法进行比较，以及探究导频长度、激活函数、学习率等参数对性能的影响

## 实验平台

- `OS`: `Manjaro Linux x86_64`
- `Kernel`: `5.4.80-1-MANJARO`
- `Python`: `3.8.6`
- `Jupyter Notebook`: `6.1.4`
- `Matplotlib`: `3.3.3`
- `Numpy`: `1.19.4`
- `Tensorflow`: `2.3.1`
- `Keras`: `2.4.0`

## 实验步骤

运行`run_all`函数，修改其参数值，分别是`CP`, `P`导频长度, `activation`激活函数, `lr`学习率，得到的结果和数据保存在`reslut`文件夹中，其中文件名后缀格式为`[CP]-[P]-[activation]-[lr]`

## 实验总结

- `P`的影响
  - `P`较大时，性能类似
  - `P`较小时，基于`DNN`的性能更好
- `CP`的影响
  - 可以克服多径干扰，需要额外的能量开销
  - 传统接收机受`CP`的影响较大， 没有`CP`，基于`DNN`的接收机有更大的增益
- 学习率的影响
  - 将学习率从调整为`0.001`调整为`0.004`，波动变大，在训练`15`次左右之后，性能随着训练次数的增加越来越差
- 激活函数的影响
  - 将激活函数从调整为`relu`调整为`sigmoid`，训练增益变小，波动更大
