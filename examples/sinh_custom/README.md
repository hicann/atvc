<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->
# SinhCustom算子样例

## 概述

样例概述：本样例介绍了如何利用ATVC实现临时Tensor参与计算的SinhCustom单算子并完成算子验证
- 算子功能：sinh
- 使用的ATVC模板：Elementwise
- 调用方式：Kernel直调


## 样例支持AI处理器型号

- Ascend 910C
- Ascend 910B


## 算子描述

SinhCustom算子数学计算公式：$y = \frac{\exp(x) - \exp(-x)}{2}$

SinhCustom算子规格：

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">SinhCustom</td></tr>

<tr><td rowspan="4" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">SinhCustom</td></tr>
</table>

## 目录结构

| 文件名                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [sinh_custom.cpp](./sinh_custom.cpp) | SinhCustom算子代码实现以及调用样例               |


## 算子运行
在ascendc-api-adv代码仓目录下执行：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh sinh_custom
```