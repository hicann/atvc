# Aclnn调用样例

## 概述
使用ATVC框架开发自定义算子，并实现Aclnn调用的样例。

## 样例支持AI处理器型号

- Ascend 910B

## 样例介绍
|  样例名称                                                   |  功能描述                                               |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [add](./add) |  使用ATVC框架开发自定义算子Add，并实现单算子API调用的样例。 |
| [reduce_sum](./reduce_sum) | 使用ATVC框架开发自定义算子ReduceSum，并实现单算子API调用的样例。 |

## 目录结构介绍
```
ops_aclnn/
├── add           // Add算子的自定义算子工程样例
└── reduce_sum    // ReduceSum算子的自定义算子工程样例
```

## 开发步骤
### 步骤1. 生成自定义工程基础目录及文件
  参考[msopgen](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0021.html)创建算子工程的基础文件。 
  ```bash 
    rm -rf CustomOp
    # Generate the op framework
    msopgen gen -i AddCustom.json -c ai_core-Ascend910B1 -lan cpp -out CustomOp
  ```
  生成目录结构如下：
  ```
    CustomOp
    ├── build.sh                  // 编译入口脚本
    ├── cmake 
    │   ├── config.cmake
    │   ├── util                  // 算子工程编译所需脚本及公共编译文件存放目录
    ├── CMakeLists.txt            // 算子工程的CMakeLists.txt
    ├── CMakePresets.json         // 编译配置项
    ├── framework                 // 算子插件实现文件目录，单算子模型文件的生成不依赖算子适配插件，无需关注
    ├── op_host                   // Host侧实现文件
    │   ├── add_custom_tiling.h   // 算子tiling定义文件
    │   ├── add_custom.cpp        // 算子原型注册、shape推导、信息库、tiling实现等内容文件
    │   ├── CMakeLists.txt
    ├── op_kernel                 // Kernel侧实现文件
    │   ├── CMakeLists.txt   
    │   ├── add_custom.cpp        // 算子代码实现文件 
    ├── scripts                   // 自定义算子工程打包相关脚本所在目录
  ```

### 步骤2. 修改对应文件内容
  
- 2.1 复制需要的配置文件

  将[func.cmake](./add/AddCustom/cmake/func.cmake)、[intf.cmake](./add/AddCustom/cmake/intf.cmake)、host侧的[CMakeLists.txt](./add/AddCustom/op_host/CMakeLists.txt)和kernel侧的[CMakeLists.txt](./add/AddCustom/op_kernel/CMakeLists.txt)分别复制到`步骤1`生成的工程文件的对应目录下。
 
- 2.2 修改对应的host文件
  - 引入对应的头文件，修改对应TilingFunc函数中tiling的生成，根据算子类型调用不同的tiling生成策略，更多ATVC的用法可参考ATVC的[开发指南](../../docs/02_developer_guide.md)。

    ElementWise类，参考[add_custom.cpp](./add/AddCustom/op_host/add_custom.cpp)
    ```cpp
      // 引入头文件
      #include "elewise/elewise_host.h"
      ...      
      //定义算子描述
      using AddOpTraitsFloat = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;
      using AddOpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int32_t, int32_t>, ATVC::OpOutputs<int32_t>>;

      // 修改对应TilingFunc
      // 声明运行态参数tiling
      ATVC::EleWiseParam *tiling = context->GetTilingData<ATVC::EleWiseParam>();
      uint32_t totleLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
      // 根据不同数据类型使用不同的算子描述
      if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_FLOAT) {
          // AddOpTraitsFloat为ADD算子描述原型，根据算子输入输出个数和实际元素数量计算出Tiling数据后填入tiling中
          (void)ATVC::Host::CalcEleWiseTiling<AddOpTraitsFloat>(totleLength, *tiling);
      } else if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_INT32) {
          (void)ATVC::Host::CalcEleWiseTiling<AddOpTraitsInt>(totleLength, *tiling);
      }
      // 设置tilingkey
      context->SetTilingKey(0);
      // 设置blockDim的大小
      context->SetBlockDim(tiling->tilingData.blockNum);
      // 设置Workspace的大小
      size_t *currentWorkspace = context->GetWorkspaceSizes(1);
      currentWorkspace[0] = 0;
      ...
    ```
   
    Broadcast类
    ```cpp
      #include "broadcast/broadcast_host.h"
      // 定义算子描述
      using BroadcastOpTraitsFloat = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
      using BroadcastOpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int32_t>, ATVC::OpOutputs<int32_t>>;
      ...
      // 修改对应TilingFunc
      // 获取输入输出shape
      std::vector<int64_t> shapeIn;
      std::vector<int64_t> shapeOut;
      for (int32_t i = 0; i < inputShape0.GetDimNum(); i++) {
          shapeIn.push_back(inputShape0.GetDim(i));
      }
      for (int32_t i = 0; i < outputShape0.GetDimNum(); i++) {
          shapeOut.push_back(outputShape0.GetDim(i));
      }
      // 声明运行态参数tiling
      ATVC::BroadcastParam *tiling = context->GetTilingData<ATVC::BroadcastParam>();
      ATVC::BroadcastPolicy policy = {-1, -1, -1};
      // 根据不同数据类型使用不同的算子描述
      if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_FLOAT) {
        // BroadcastOpTraitsFloat为Reduce算子描述原型，根据算子输入shape和dim计算出Tiling数据后填入tiling中
        (void)ATVC::Host::CalcBroadcastTiling<BroadcastOpTraitsFloat>(shapeIn, shapeOut, &policy, tiling);
      } else if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_INT32) {
        (void)ATVC::Host::CalcBroadcastTiling<BroadcastOpTraitsInt>(shapeIn, shapeOut, &policy, tiling);
      }
      // 根据不同的policy设置不同的tilingkey,在kernel侧根据不同的tilingkey进行调用不同的算子模版
      if (policy == ATVC::BROADCAST_POLICY0) {
        context->SetTilingKey(0);
      } else if (policy == ATVC::BROADCAST_POLICY1) {
        context->SetTilingKey(1);
      }
      // 设置blockDim
      context->SetBlockDim(tiling->tilingData.coreNum);
      size_t *currentWorkspace = context->GetWorkspaceSizes(1);
      currentWorkspace[0] = 0;
    ```

    reduce_sum类，参考[reduce_sum_custom.cpp](./reduce_sum/ReduceSumCustom/op_host/reduce_sum_custom.cpp)
    ```cpp
      // 引入头文件
      #include "reduce/reduce_host.h"
      // 定义算子描述
      using ReduceOpTraitsFloat = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
      using ReduceOpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int32_t>, ATVC::OpOutputs<int32_t>>;
      ...
      // 修改对应TilingFunc
      ATVC::ReducePolicy policy = {0, 0, 0};
      auto inputShape0 = context->GetInputShape(0)->GetOriginShape();
      std::vector<int64_t> shapeIn;
      for (int32_t i = 0; i < inputShape0.GetDimNum(); i++) {
          shapeIn.push_back(inputShape0.GetDim(i));
      }
      // 获取dim值
      const gert::RuntimeAttrs *runtimeAttrs = context->GetAttrs();
      const gert::TypedContinuousVector<int64_t> *attr0 = runtimeAttrs->GetListInt(0);
      const int64_t *arr = reinterpret_cast<const int64_t *>(attr0->GetData());
      std::vector<int64_t> dim(arr, arr + attr0->GetSize());

      // 声明运行态参数tiling
      ATVC::ReduceParam *tiling = context->GetTilingData<ATVC::ReduceParam>();
      tiling->nBufferNum = 2;
      // 根据不同数据类型使用不同的算子描述
      if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_FLOAT) {
        // ReduceOpTraitsFloat为Reduce算子描述原型，根据算子输入shape和dim计算出Tiling数据后填入tiling中
        (void)ATVC::Host::CalcReduceTiling<ReduceOpTraitsFloat>(shapeIn, dim, &policy, tiling);
      } else if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_INT32) {
        (void)ATVC::Host::CalcReduceTiling<ReduceOpTraitsInt>(shapeIn, dim, &policy, tiling);
      }
      // 设置policyId，作为kernel的分支判断
      tiling->policyId = policy.getID();
      // 设置blockDim
      context->SetBlockDim(tiling->tilingData.coreNum);
    ```
 - 2.3 修改对应的kernel文件
  
    用户需要通过AscendC API来搭建Add算子的核心计算逻辑，在ATVC框架中，这类算子的核心计算逻辑是通过定义一个结构体的仿函数来实现。它需要`ATVC::OpTraits`作为固定模板参数，并重载`operator()`来被提供的Kernel层算子模板类调用，更多ATVC的用法可参考atvc的[开发指南](../../docs/02_developer_guide.md)。
 
    ElementWise类[add_custom.cpp](./add/AddCustom/op_kernel/add_custom.cpp)
    ```cpp
      // 头文件引入
      #include "elewise/elewise_device.h"
      ...
      // 定义算子描述
      using AddOpTraits = ATVC::OpTraits<ATVC::OpInputs<DTYPE_X, DTYPE_Y>, ATVC::OpOutputs<DTYPE_Z>>;

      ...
      // 新增 AddComputeFunc
      // 传入编译态参数ATVC::OpTraits
      template<typename Traits>
      struct AddComputeFunc {
          /*
          函数说明： z = x + y
          参数说明：
              x                   : 参与运算的输入
              y                   : 参与运算的输入
              z                   : 参与运算的输出
          */
          template<typename T> 
          // 重载operator，提供给算子模板类调用
          __aicore__ inline void operator()(AscendC::LocalTensor<T> x, AscendC::LocalTensor<T> y, AscendC::LocalTensor<T> z) {
              AscendC::Add(z, x, y, z.GetSize()); // 开发调用AscendC Api自行实现计算逻辑, 通过z.GetSize()获取单次计算的元素数量
          }
      };

      // 修改核函数文件的实现
      KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
      REGISTER_TILING_DEFAULT(ATVC::EleWiseParam);
      GET_TILING_DATA(param, tiling);
      auto op = ATVC::Kernel::EleWiseOpTemplate<AddComputeFunc<AddOpTraits>>();
      op.Run(x, y, z, &param);
    ```    
    Broadcast类
    ```cpp
      // 头文件引入
      #include "broadcast/broadcast_device.h"
      // 定义算子描述
      using BroadcastOpTraits = ATVC::OpTraits<ATVC::OpInputs<DTYPE_X>, ATVC::OpOutputs<DTYPE_Y>>;
      ...

      // 修改核函数文件
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        REGISTER_TILING_DEFAULT(ATVC::BroadcastParam);
        GET_TILING_DATA(tilingData, tiling);
        // broadcast有不同的policy，在host与tilingkey进行绑定，此处的调用使用TILING_KEY_IS进行判断，和host的文件SetTilingKey相对应
        if (TILING_KEY_IS(0)) {
          auto op = ATVC::Kernel::BroadcastOpTemplate<ATVC::BroadcastCompute<BroadcastOpTraits>, ATVC::BROADCAST_POLICY0>();
          ATVC::BroadcastParam param = &tilingData;
          op.Run(x, y, &param);
        }else{
          ...
        }
    ```
    reduce_sum类[reduce_sum_custom.cpp](./reduce_sum/ReduceSumCustom/op_kernel/reduce_sum_custom.cpp)
    ```cpp
      // 头文件引入
      #include "reduce/reduce_device.h"
      // 定义算子描述
      using ReduceOpTraits = ATVC::OpTraits<ATVC::OpInputs<DTYPE_X>, ATVC::OpOutputs<DTYPE_Y>>;
      ...

      // 修改核函数文件
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        REGISTER_TILING_DEFAULT(ATVC::ReduceParam);         
        GET_TILING_DATA(param, tiling);
        if (param.policyId == ATVC::REDUCE_POLICY0.ID) {
            auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY0>();
            op.Run(x, y, &param);
        } else {
          // 根据不同的tiling.policyId进行判断不同ReduceOpTemplate初始化
           ...
        }
    ```
    此处未使用`TILING_KEY_IS`进行分支判断，因为policy分支过多，在使用`tilingKey`进行判断的时候，会有爆栈的问题，此时建议使用`param.policyId`进行分支的判断。

### 步骤3. 算子工程编译

  在算子工程目录下执行如下命令，进行算子工程编译。  
  ```bash
  cd CustomOp
  bash build.sh
  ```
脚本运行成功后，会在当前目录下创建CustomOp目录，编译完成后，会在CustomOp/build_out中，生成自定义算子安装包custom_opp_<target os>_<target architecture>.run，例如“custom_opp_ubuntu_x86_64.run”。

### 步骤4. 部署自定义算子包

- 部署自定义算子包前，请确保存在自定义算子包默认部署路径环境变量ASCEND_OPP_PATH。

  ```bash 
  echo $ASCEND_OPP_PATH
  # 输出示例 /usr/local/Ascend/latest/opp

  # 若没有，则需导出CANN环境变量
  source [ASCEND_INSTALL_PATH]/bin/setenv.bash
  # 例如 source /usr/local/Ascend/latest/bin/setenv.bash
  ```
  其中ASCEND_INSTALL_PATH为CANN软件包安装路径，一般和上一步中指定的路径保持一致。  
- 在自定义算子安装包所在路径下，执行如下命令安装自定义算子包。命令执行成功后，自定义算子包中的相关文件将部署至opp算子库环境变量ASCEND_OPP_PATH指向的的vendors/customize目录中。

  ```bash
  cd build_out
  ./custom_opp_<target os>_<target architecture>.run
  ```

### 步骤5. 调用执行算子工程 

  算子文件编写完成，参考[aclnn调用AddCustom算子工程（代码简化）](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/1_add_frameworklaunch/AclNNInvocationNaive/README.md)进行编译验证。