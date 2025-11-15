# PyTorch调用样例

## 概述
使用ATVC开发自定义算子，并实现从PyTorch框架调用的样例。

## 样例介绍
|  样例名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [add](./add) | 基于ATVC框架的Add自定义Vector算子 |
| [reduce_sum](./reduce_sum) | 基于ATVC框架的reduce_sum自定义Vector算子 | 

## 目录结构介绍
```
ops_pytorch/
├── add         // Add算子的PyTorch调用样例
└── reduce_sum  // ReduceSum算子的PyTorch调用样例
```

## 开发步骤

  不同的算子类型可参考[快速入门](../../docs/01_quick_start.md)中的模版选择模版进行选择，用户在此处通过`<<<>>>`的方式调用核函数，更多ATVC的用法可参考ATVC的[开发指南](../../docs/02_developer_guide.md)。
  
### 步骤1. 引入头文件。需要注意的是，需要保护对应核函数调用接口声明所在的头文件{kernel_name}_impl.h,kernel_name为算子的核函数名称。
```cpp
    // 头文件引入
    #include <torch/extension.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "add_custom_impl.h"
```
### 步骤2. 应用程序框架编写，需要注意的是，本样例输入x，y的内存是在python调用脚本[run_op.py](./add/run_op.py)中分配的。
```cpp
    namespace ascendc_elewise_ops {
    at::Tensor op_add_custom(const at::Tensor &x, const at::Tensor &y) {
    }
    }
```
### 步骤3. NPU侧运行验证。通过`<<<>>>`的方式调用核函数完成指定的运算。
```cpp
    // 运行资源申请，通过c10_npu::getCurrentNPUStream()的函数获取当前NPU上的流
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // 分配Device侧输出内存
    at::Tensor z = at::empty_like(x);
    // totalLength是算子输入的元素个数
    int32_t totalLength = 1;
    for (int32_t size : x.sizes()) {
        totalLength *= size;
    }
    // 声明运行态参数param
    ATVC::EleWiseParam param;
    // Host侧调用Tiling API完成相关运行态参数的运算
    (void)ATVC::Host::CalcEleWiseTiling<AddOpTraitsFloat>(totalLength, param);
    // 使用<<<>>方式调用核函数完成指定的运算
    AddCustom<AddOpTraitsFloat><<<param.tilingData.blockNum, nullptr, stream>>>(
        (uint8_t *)(x.storage().data()), (uint8_t *)(y.storage().data()), (uint8_t *)(z.storage().data()), param);
    // 将Device上的运算结果拷贝回Host并释放申请的资源
    return z;
```
### 步骤4. 定义PyTorch算子的调用接口。
``` cpp 
    // 加载算子模版
    TORCH_LIBRARY(ascendc_ops, m) {  // 模块名ascendc_ops，模块对象m
        m.def("add", &ascendc_elewise_ops::op_add_custom); // 将函数add和PyTorch进行绑定
    }
       
``` 
