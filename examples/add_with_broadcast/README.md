<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->
# AddWithBroadcast算子样例

## 概述

样例概述：本样例介绍了利用ATVC实现带广播的Add单算子并完成功能验证
- 算子功能：add
- 使用的ATVC模板：带后置Elementwise计算的Broadcast模板
- 调用方式：Kernel直调


## 样例支持AI处理器型号：
- Ascend 910C
- Ascend 910B

## 算子描述

算子数学计算公式：$z = x + y$

算子规格：

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>

<tr><td rowspan="4" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">1 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">AddWithBroadcastCustom</td></tr>
</table>

## 目录结构

| 文件名                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [add_with_broadcast.cpp](./add_with_broadcast.cpp) | AddWithBroadcast算子代码实现以及调用样例               |
| [add_with_broadcast.h](./add_with_broadcast.h) | AddWithBroadcast算子代码实现头文件           |
| [post_compute_add_of_broadcast.h](./post_compute_add_of_broadcast.h) | 后置Elementwise计算              |


## 算子运行
在ascendc-api-adv代码仓目录下执行：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh add_with_broadcast
```