# Add算子样例

## 概述
本样例基于AddCustom算子工程，介绍了基于ATVC的PyTorch工程及调用。

## 目录结构介绍
```
├── add                                   
│   ├── add_custom_impl.h                 // 通过PyTorch调用的方式调用Add算子
│   ├── pytorch_ascendc_extension.cpp     // PyTorch调用入口
│   ├── run_op.py                         // PyTorch的测试用例
│   └── run.sh                            // 脚本，编译需要的二进制文件，并测试
```

## 算子描述
Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：
```
z = x + y
```
## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">int32_t、float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">int32_t、float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">int32_t、float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">AddCustom</td></tr>
</table>

## 编译运行样例算子
针对PyTorch算子，编译运行包含如下步骤：
- 完成算子PyTorch入口和impl文件的实现；
- 编译PyTorch算子的二进制文件；
- 调用执行PyTorch算子。

详细操作如下所示。
### 1. 获取源码包及环境配置
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包及环境变量的准备。
### 2. 安装PyTorch环境
参考[torch的安装](https://gitee.com/ascend/pytorch)进行安装torch、torch_npu环境

### 3. 基于ATVC编写PyTorch算子的实现<a name="operatorcompile"></a>
  - 算子kernel侧实现
  
    编写kernel侧函数，完成指定的运算。参考[add_custom_impl.h](./add_custom_impl.h)和[开发指南](../../../docs/02_developer_guide.md)完成核函数的实现。 
    
  - 编写PyTorch入口函数，并通过`<<<>>>`调用核函数，参考[pytorch_ascendc_extension.cpp](./pytorch_ascendc_extension.cpp)
    ```cpp
    at::Tensor op_add_custom(const at::Tensor &x, const at::Tensor &y)
    {
        // 运行资源申请，通过c10_npu::getCurrentNPUStream()的函数获取当前NPU上的流
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        // 分配Device侧输出内存
        at::Tensor z = at::empty_like(x);
        ATVC::EleWiseParam param;
        int32_t totalLength = 1;
        for (int32_t size : x.sizes()) {
            totalLength *= size;
        }
        (void)ATVC::Host::CalcEleWiseTiling<AddOpTraitsFloat>(totalLength, param);
        // 使用<<<>>方式调用核函数完成指定的运算
        AddCustom<AddOpTraitsInt><<<param.tilingData.blockNum, nullptr, stream>>>(
            (uint8_t *)(x.storage().data()), (uint8_t *)(y.storage().data()), (uint8_t *)(z.storage().data()), param);
        return z;
    }
    ```
  - 编写python调用函数,并调用pytorch入口函数，参考[run_op.py](./run_op.py)

    ```py
    # 引入头文件
    import torch
    import torch_npu
    import numpy as np
    from torch_npu.testing.testcase import TestCase, run_tests
    # 加载二进制文件
    torch.npu.config.allow_internal_format = False
    torch.ops.load_library('./libascendc_pytorch.so')

    class TestAscendCOps(TestCase):
      # 测试用例
      def test_add_custom_ops_float(self):
          # 分配Host侧输入内存，并进行数据的初始化
          length = [8, 2048]    
          x = torch.rand(length, device='cpu', dtype=torch.float32)
          y = torch.rand(length, device='cpu', dtype=torch.float32)
          # 分配Device侧内存，并将数据从Host上拷贝到Device上
          npuout = torch.ops.ascendc_ops.add(x.npu(), y.npu())
          cpuout = torch.add(x, y)
          self.assertRtolEqual(npuout, cpuout)

    # 测试用例调用    
    if __name__ == '__main__':
      run_tests()
      ```
  - 编译和测试脚本，参考[run.sh](./run.sh)

    ```sh
    # 获取torch、torch_npu、python的lib和include路径和atvc的路径       
    torch_location=...
    torch_npu_location=...
    python_include=...
    python_lib=... 
    if [ -z "$ATVC_PATH" ]; then
        atvc_path=$(realpath ../../../include)
    else
        atvc_path=$ATVC_PATH
    fi

    # 使用bisheng进行编译PyTorch算子
    bisheng -x cce pytorch_ascendc_extension.cpp \
    -D_GLIBCXX_USE_CXX11_ABI=0  \
    -I${torch_location}/include   \
    -I${torch_location}/include/torch/csrc/api/include   \
    -I${python_include}   \
    -I${atvc_path}   \
    -I${torch_npu_location}/include   \
    -L${torch_location}/lib   \
    -L${torch_npu_location}/lib   \
    -L${python_lib}   \
    -L${lib_path}   \
    -L${_ASCEND_INSTALL_PATH}/lib64 \
    -ltorch -ltorch_cpu -lc10 -ltorch_npu -lpython3 -ltorch_python   \
    -shared -cce-enable-plugin --cce-aicore-arch=dav-c220 -fPIC -ltiling_api -lplatform -lm -ldl  \
    -o libascendc_pytorch.so

    # 执行测试用例
    python3 run_op.py
    ```

### 4. 基于ATVC编写PyTorch算子的调用验证
  - 导入ATVC环境变量
    ```bash
    # 如果不导入，默认使用./atvc/include路径
    export ATVC_PATH=${atvc}/include
    ```
  - 调用脚本，生成PyTorch算子，并运行测试用例 
      ```bash
      cd ./ops_templates/atvc/examples/ops_pytorch/add
      bash run.sh
      ...
      OK
      ```    
