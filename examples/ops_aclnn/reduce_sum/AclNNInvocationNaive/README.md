# AclNNInvocationNaive工程样例

## 概述
本样例相比于AclNNInvocation样例工程，简化了工程配置。
## 目录结构介绍
```
├── AclNNInvocationNaive
│   ├── CMakeLists.txt      // 编译规则文件
│   ├── main.cpp            // 单算子调用应用的入口
│   └── run.sh              // 编译运行算子的脚本
```
## 代码实现介绍
完成自定义算子的开发部署后，可以通过单算子调用的方式来验证单算子的功能。main.cpp代码为单算子API执行方式。单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。

自定义算子编译部署后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，形如：
   ```cpp
   // 获取算子使用的workspace空间大小
   aclnnStatus aclnnReduceSumCustomGetWorkspaceSize(const aclTensor *x, const aclIntArrat *dim, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
   // 执行算子
   aclnnStatus aclnnReduceSumCustom(void *workspace, int64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
   ```
其中`aclnnReduceSumCustomGetWorkspaceSize`为第一段接口，主要用于计算本次API调用计算过程中需要多少的`workspace`内存。获取到本次API计算需要的`workspace`大小之后，开发者按照`workspaceSize`大小申请Device侧内存，然后调用第二段接口`aclnnReduceSumCustom`执行计算。具体参考[单算子API调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)章节。
## 运行样例算子
### 1. 编译算子工程
运行此样例前，请参考[编译算子工程](../README.md#operatorcompile)完成前期准备。
### 2. aclnn调用样例运行

  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd atvc/examples/ops_aclnn/reduce_sum/AclNNInvocationNaive
    ```
  - 样例编译文件修改

    将CMakeLists.txt文件内"/usr/local/Ascend/ascend-toolkit/latest"替换为CANN软件包安装后的实际路径。  
    eg:/home/HwHiAiUser/Ascend/ascend-toolkit/latest

  - 样例执行

    用户参考run.sh脚本进行编译与运行。
    ```bash
    bash run.sh
    ```
