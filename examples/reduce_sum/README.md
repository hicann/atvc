<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->
# ReduceSum算子样例

## 概述

样例概述：本样例介绍了利用ATVC实现ReduceSum单算子并完成功能验证
- 算子功能：对输入tensor的指定轴进行规约累加的计算并输出结果
- 使用的ATVC模板：Reduce
- 调用方式：Kernel直调


## 样例支持AI处理器型号

- Ascend 910C
- Ascend 910B


## 算子描述

ReduceSum是对输入tensor的指定轴进行规约累加的计算并输出结果的Reduce类算子。

ReduceSum算子规格：

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">ReduceSum</td></tr>

<tr><td rowspan="4" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">y</td><td align="center">1 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ReduceCustom</td></tr>
</table>

## 目录结构

| 文件名                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [reduce_sum.cpp](./reduce_sum.cpp) | ReduceSum算子代码实现以及调用样例               |


## 算子运行
在ascendc-api-adv代码仓目录下执行：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh reduce_sum
```