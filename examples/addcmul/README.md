<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->
# Addcmul算子样例


## 概述

样例概述：本样例介绍了利用ATVC实现Addcmul单算子并完成功能验证
- 算子功能：Addcmul
- 使用的ATVC模板：带前置Elementwise计算和后置Elementwise计算的Broadcast模板
- 调用方式：Kernel直调

## 样例支持AI处理器型号：
- Ascend 910C
- Ascend 910B

## 算子描述

算子数学计算公式：$output_i = input_i + value * tensor1_i * tensor2_i$

算子规格：

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>

<tr><td rowspan="5" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">input</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">tensor1</td><td align="center">1 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">tensor2</td><td align="center">1 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">output</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">AddcmulCustom</td></tr>
</table>

## 目录结构

| 文件名                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [addcmul.cpp](./addcmul.cpp) | Addcmul算子代码实现以及调用样例               |
| [addcmul.h](./addcmul.h) |      Addcmul算子代码实现头文件      |
| [post_compute_add_of_broadcast.h](./post_compute_add_of_broadcast.h) |        后置Elementwise计算        |
| [pre_compute_mul_of_broadcast.h)](./pre_compute_mul_of_broadcast.h) |        前置Elementwise计算        |


## 算子运行
在ascendc-api-adv代码仓目录下执行：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh addcmul
```