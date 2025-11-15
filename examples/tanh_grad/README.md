<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->
# TanhGrad算子样例

## 概述

样例概述：本样例介绍了利用ATVC实现TanhGrad单算子并验证了调试调优相关功能验证。
- 算子功能：TanhGrad
- 使用的ATVC模板：Elementwise
- 调用方式：Kernel直调

## 样例支持AI处理器型号

- Ascend 910C
- Ascend 910B

## 算子描述

算子数学计算公式：$z = dy * (1 - y ^ 2)$

算子规格：

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">TanhGrad</td></tr>

<tr><td rowspan="4" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">dy</td><td align="center">8 * 1024</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">8 * 1024</td><td align="center">float</td><td align="center">ND</td></tr>
<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">z</td><td align="center">8 * 1024</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">TanhGrad</td></tr>
</table>

## 目录结构

| 文件名                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [tanh_grad.cpp](./tanh_grad.cpp) | Tanh算子代码实现以及调用样例               |


## 算子基本功能验证
执行命令如下：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh tanh_grad
```

## 算子调试调优
样例提供的主要调试调优方式如下：
- 使用`ATVC::Host::EleWiseTilingHyperParam`构建超参对`ATVC::Host::CalcEleWiseTiling()`接口实现Tiling调优
- 使用`--run-mode=debug_print`进行DFX信息打印：
执行命令如下：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh tanh_grad --run-mode=debug_print
```

- 使用`--run-mode=profiling`开启Profiling，获取性能数据：
执行命令如下：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh tanh_grad --run-mode=profiling
```

更多详细的调试调优介绍参考[ATVC开发指南](../../docs/02_developer_guide.md)的`ATVC的调试调优功能`章节