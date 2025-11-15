# ReduceSum算子样例

## 概述
本样例基于ReduceSumCustom算子工程，介绍了基于ATVC的自定义算子工程调用。

## 目录结构介绍
```
├── reduce_sum                 // 使用框架调用的方式调用ReduceSum算子
│   ├── AclNNInvocation        // 通过aclnn调用的方式调用ReduceSumCustom算子
│   ├── ReduceSumCustom        // ReduceSumCustom算子工程
│   ├── ReduceSumCustom.json   // ReduceSumCustom算子的原型定义json文件
│   └── install.sh             // 脚本，调用msOpGen生成自定义算子工程，并编译
```

## 算子描述
ReduceSum是对输入tensor的指定轴进行规约累加的计算并输出结果的Reduce类算子。

## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">ReduceSum</td></tr>
</tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
 
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reduce_sum_custom</td></tr>
</table>

## 算子工程介绍
其中，算子工程目录ReduceSumCustom包含算子的实现文件，如下所示：
```
├── ReduceSumCustom           // ReduceSumCustom自定义算子工程
│   ├── op_host               // host侧实现文件
│   ├── op_kernel             // kernel侧实现文件
│   ├── build.sh              // 算子构建入口
│   └── CMakeLists.txt        // 算子的cmake文件
```
CANN软件包中提供了工程创建工具msOpGen，ReduceSumCustom算子工程可通过ReduceSumCustom.json自动创建，自定义算子工程具体请参考[Ascend C算子开发](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)>工程化算子开发>创建算子工程 章节。

创建完自定义算子工程后，开发者重点需要完成算子host和kernel文件的功能开发。为简化样例运行流程，本样例已在ReduceSumCustom目录中准备好了必要的算子实现，install.sh脚本会创建一个CustomOp目录，并将算子实现文件复制到对应目录下，再编译算子。

备注：CustomOp目录为生成目录，每次执行install.sh脚本都会删除该目录并重新生成，切勿在该目录下编码算子，会存在丢失风险。

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 调用msOpGen工具生成自定义算子工程；
- 基于ATVC框架完成算子host和kernel实现；
- 编译自定义算子工程生成自定义算子包；
- 安装自定义算子包到自定义算子库中；
- 调用执行自定义算子。

详细操作如下所示。
### 1. 获取源码包及环境配置
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包及环境变量的准备。

### 2. 生成自定义算子工程，复制host和kernel实现并编译算子<a name="operatorcompile"></a>
  - 导入ATVC环境变量
    ```bash
    # 如果不导入，默认使用./atvc/include路径
    export ATVC_PATH=${atvc}/include
    ```
  - 切换到msOpGen脚本install.sh所在目录
    ```bash
    # 若开发者以git命令行方式clone了master分支代码，并切换目录
    cd ./ops_templates/atvc/examples/ops_aclnn/reduce_sum
    ```

  - 调用脚本，生成自定义算子工程，复制host和kernel实现并编译算子
 
      运行install.sh脚本
      ```bash
      bash install.sh -v [SOC_VERSION]
      ``` 
    参数说明：
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下处理器型号：
        - Ascend 910C
        - Ascend 910B

    脚本运行成功后，会在当前目录下创建CustomOp目录，编译完成后，会在CustomOp/build_out中，生成自定义算子安装包custom_opp_\<target os>_\<target architecture>.run，例如“custom_opp_ubuntu_x86_64.run”。

### 3. 部署自定义算子包
- 部署自定义算子包前，请确保存在自定义算子包默认部署路径环境变量ASCEND_OPP_PATH。
    ```bash
    echo $ASCEND_OPP_PATH
    # 输出示例 /usr/local/Ascend/latest/opp

    # 若没有，则需导出CANN环境变量
    source [ASCEND_INSTALL_PATH]/bin/setenv.bash
    # 例如 source /usr/local/Ascend/latest/bin/setenv.bash
    ```
    参数说明：
    - ASCEND_INSTALL_PATH：CANN软件包安装路径，一般和上一步中指定的路径保持一致

- 在自定义算子安装包所在路径下，执行如下命令安装自定义算子包
    ```bash
    cd CustomOp/build_out
    ./custom_opp_<target os>_<target architecture>.run
    ```
  命令执行成功后，自定义算子包中的相关文件将部署至opp算子库环境变量ASCEND_OPP_PATH指向的的vendors/customize目录中。

### 4. 调用执行算子工程
- [aclnn调用ReduceSumCustom算子工程](./AclNNInvocationNaive/README.md)
