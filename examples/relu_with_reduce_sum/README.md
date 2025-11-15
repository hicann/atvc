<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->
# ReluWithReduceSum算子样例

## 概述

样例概述：本样例介绍了利用ATVC实现ReduceSum+Elementwise算子并完成功能验证，此算子精度适用遵循reduce的精度范围。
- 算子功能：ReduceSum算子前置添加Add和后置添加Relu自定义算子
- 使用的ATVC模板：带前后Elementwise计算的Reduce模板
- 调用方式：Kernel直调


## 样例支持AI处理器型号

- Ascend 910C
- Ascend 910B

## 算子描述

该自定义算子数学计算公式为：
$$ 

y = \begin{cases} 0& \text{if reduceSum(x + 1) < 0}\\
reduceSum(x + 1)& \text{if reduceSum(x + 1) ≥ 0}
\end{cases}
$$

算子规格：

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">ReluWithReduceSum</td></tr>

<tr><td rowspan="4" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8 * 1024</td><td align="center">float</td><td align="center">ND</td></tr>
<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">y</td><td align="center">1 * 1024</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ReluWithReduceSum</td></tr>
</table>

## 目录结构

| 文件名                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [relu_with_reduce_sum.cpp](./relu_with_reduce_sum.cpp) | 算子代码实现以及调用样例               |
| [relu_with_reduce_sum.h](./relu_with_reduce_sum.h) | 算子代码实现头文件           |
| [pre_compute_add_with_reduce_sum.h](./pre_compute_add_with_reduce_sum.h) | 前置Elementwise计算              |
| [post_compute_relu_with_reduce_sum.h](./post_compute_relu_with_reduce_sum.h) | 后置Elementwise计算              |

## 算子运行
在ascendc-api-adv代码仓目录下执行：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh relu_with_reduce_sum
```