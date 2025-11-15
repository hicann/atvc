/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include "acl/acl.h"
#include "broadcast/host/broadcast_host.h"
#include "broadcast/kernel/broadcast_kernel.h"
#include "example_common.h"

namespace {
void CleanUp(uint8_t *&xDevice, uint8_t *&yDevice, uint8_t *&yHost)
{
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(yHost));
}

// BroadcastTo算子的描述：一个输入，一个输出，类型均为float
using BroadcastOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
}
/*
 * 该函数为BroadcastCustom算子核函数入口
 * x                 Device上的gm地址，指向BroadcastCustom算子第一个输入
 * y                 Device上的gm地址，指向BroadcastCustom算子第一个输出
 * broadcastParam    指向运行态ATVC::BroadcastParam数据
*/
template<typename Traits, const auto& Policy>
__global__ __aicore__ void BroadcastCustom(GM_ADDR x, GM_ADDR y, ATVC::BroadcastParam broadcastParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 将计算模板类模板定义作为模板参数传入，Policy由Host层的策略分派API给出
    auto op = ATVC::Kernel::BroadcastOpTemplate<ATVC::BroadcastCompute<Traits>, Policy>();
    ATVC::BroadcastParam *param = &broadcastParam;
    op.Run(x, y, param);
}
namespace {
// 负责Broadcast类算子的调度，选择对应的Policy最佳策略并执行Kernel函数
template<class OpTraits>
void BroadcastOpAdapter(uint8_t* x, uint8_t* y, ATVC::BroadcastParam &param, ATVC::BroadcastPolicy &policy, aclrtStream& stream)
{
    // 申请临时空间workspace
    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, param.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    param.workspaceAddr = reinterpret_cast<uint64_t>(workspaceDevice);
    // 将tiling api计算出的BroadcastPolicy转化为编译态参数并实例化相应的核函数
    if (policy == ATVC::BROADCAST_POLICY0) {
        BroadcastCustom<OpTraits, ATVC::BROADCAST_POLICY0><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
     }else if (policy == ATVC::BROADCAST_POLICY1) {
        BroadcastCustom<OpTraits, ATVC::BROADCAST_POLICY1><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else {
        printf("[ERROR]: Cannot find any matched policy.\n");
    }
    // 流同步后释放申请的param内存
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtFree(workspaceDevice));
}
}

int32_t main(int32_t argc, char* argv[])
{
    if (!ATVC::Host::DebugCheck<BroadcastOpTraits, ATVC::TemplateType::BROADCAST>()) {
        printf("[ERROR]: Broadcast opTraits check failed.\n");
        return -1;
    }
    int32_t eleNum = 1 * 1024;
    int32_t outEleNum = 8 * 1024;
    size_t inputByteSize = static_cast<size_t>(eleNum) * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(outEleNum) * sizeof(float);
    std::vector<int64_t> shapeIn{1, 1024};    // 测试输入shape
    std::vector<int64_t> shapeOut{8, 1024};    // 测试输入shape
    std::vector<float> inputX(eleNum, 1.0f);
    std::vector<float> golden(outEleNum, 1.0f);

    // 初始化Acl资源
    aclrtContext context;
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    InitializeACL(context, stream, deviceId);
    uint8_t *yHost;
    uint8_t *xDevice;
    uint8_t *yDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&yHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, inputX.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ATVC::BroadcastParam param;    // Broadcast运行态参数，包含TilingData以及临时空间的相关信息
    ATVC::BroadcastPolicy policy = {-1, -1, -1};  // Broadcast运行态参数，负责映射最适合的Broadcast模板实现
    // Host侧调用Tiling API完成相关运行态参数的运算
    if (!ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param)) {
        printf("Broadcast tiling error.\n");
        return -1;
    };

    // 调用Adapter调度接口，完成核函数的模板调用
    BroadcastOpAdapter<BroadcastOpTraits>(xDevice, yDevice, param, policy, stream);

    CHECK_ACL(aclrtMemcpy(yHost, outputByteSize, yDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> outputY(reinterpret_cast<float*>(yHost), reinterpret_cast<float*>(yHost) + outEleNum);

    // 释放Acl资源
    CleanUp(xDevice, yDevice, yHost);
    CleanACL(stream, context, deviceId);

    for (int32_t i = 0; i < outEleNum; i++) {
        if (!IsClose(golden[i], outputY[i])) {
            printf("Accuracy verification failed! The expected value of element in index [%d] is %f, but actual value is %f.\n", i, golden[i], outputY[i]);
            return -1;
        }
    }
    if (!VerifyResults(golden, outputY)) {
        return -1;
    }
    printf("Accuracy verification passed.\n");
    return 0;
}
