<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->
# Edge算子样例

## 概述

样例概述：本样例介绍了利用ATVC实现自定义Edge单算子并完成功能验证
- 算子功能：自定义Edge计算的功能(一个元素的结果为周围相邻元素通过自定义计算得到的结果)
- 使用的ATVC模板：Pool
- 调用方式：Kernel直调


## 样例支持AI处理器型号：
- Ascend 910C
- Ascend 910B

## 算子描述

自定义Edge算子数学计算公式：
```
输入为二维数组：例如
x = [ x0, x1, x2, ...
      x3, x4, x5, ...
      x6, x7, x8, ...]

y4 = min(abs(((x2 + x5 + x8) - (x0 + x3 + x6)) / 3), 255)
以此类推其他元素的计算结果。
```


自定义Edge算子规格：

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Edge</td></tr>

<tr><td rowspan="4" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">width</td><td align="center">height</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">1023</td><td align="center">2517</td><td align="center">float</td><td align="center">ND</td></tr>

<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">z</td><td align="center">1023</td><td align="center">2517</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">EdgeCustom</td></tr>
<tr><td rowspan="1" align="center">规格限制说明</td><td colspan="5" align="center">当前模板只支持2维shape按16元素个数对齐、 TILE_LAYOUT{16, 16}、TILE_PADDING{8, 8, 1, 1}的场景</td></tr>
</table>

## 目录结构

| 文件名                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [edge.cpp](./edge.cpp) | 自定义Edge算子代码实现以及调用样例               |

## 算子运行
在ascendc-api-adv代码仓目录下执行：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh edge
```
当前`PoolOpTemplate`暂不支持ATVC调试调优功能，相关功能待后续补充。