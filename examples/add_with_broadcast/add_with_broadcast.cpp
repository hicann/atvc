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
#include "broadcast/host/broadcast_host.h"
#include "example_common.h"
#include "acl/acl.h"
#include "add_with_broadcast.h"

namespace {
// AddWithBroadcast算子的描述：两个输入，一个输出，类型均为float
using BroadcastOpTraits = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>, ATVC::OpTemps<float>>;

// 负责Broadcast类算子的调度，选择对应的Policy最佳策略并执行Kernel函数
template<class OpTraits>
void BroadcastOpAdapter(uint8_t* x, uint8_t* y, uint8_t* z, ATVC::BroadcastParam &param, ATVC::BroadcastPolicy &policy)
{
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    // 申请临时空间workspace，并将其与BroadcastTilingData一同传到Device侧
    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, param.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    param.workspaceAddr = reinterpret_cast<uint64_t>(workspaceDevice);
    // 将tiling api计算出的BroadcastPolicy转化为编译态参数并实例化相应的核函数
    if (policy == ATVC::BROADCAST_POLICY0) {
        AddWithBroadcastCustom<OpTraits, ATVC::BROADCAST_POLICY0><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, z, param);
     }else if (policy == ATVC::BROADCAST_POLICY1) {
        AddWithBroadcastCustom<OpTraits, ATVC::BROADCAST_POLICY1><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, z, param);
    } else {
        printf("[ERROR] Cannot find any matched policy.\n");
    }
    // 流同步后释放申请的param内存
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtFree(workspaceDevice));
}

void InitializeData(int32_t eleNum, int32_t outEleNum, std::vector<float> &inputX, std::vector<float> &inputY,
    std::vector<float> &golden)
{
    if (eleNum == 0) {
        return;
    }
    std::random_device rd;
    std::mt19937 gen(static_cast<std::mt19937::result_type>(rd()));
    std::uniform_real_distribution<float> disX(1.0f, 9.0f);
    std::uniform_real_distribution<float> disY(1.0f, 9.0f);

    for (int i = 0; i < eleNum; ++i) {
        inputX[i] = (disX(gen));
    }
    for (int i = 0; i < outEleNum; ++i) {
        inputY[i] = (disY(gen));
    }
    for (int i = 0; i < outEleNum; ++i) {
        golden[i] = (inputX[i % eleNum]) + (inputY[i]);
    }
}

void CleanUp(uint8_t *&xDevice, uint8_t *&yDevice, uint8_t *&zDevice, uint8_t *&zHost)
{
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(zHost));
}
}


int32_t main(int32_t argc, char* argv[])
{
    int32_t eleNum = 1 * 2048;
    int32_t outEleNum = 8 * 2048;
    std::vector<int64_t> shapeIn{1, 2048};    // 测试输入shape
    std::vector<int64_t> shapeOut{8, 2048};    // 测试输入shape

    size_t inputByteSize = static_cast<size_t>(eleNum) * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(outEleNum) * sizeof(float);

    std::vector<float> inputX(eleNum);
    std::vector<float> inputY(outEleNum);
    std::vector<float> golden(outEleNum);

    // 初始化Acl资源
    aclrtContext context;
    aclrtStream stream = nullptr;
    int32_t deviceId = 0;
    InitializeACL(context, stream, deviceId);
    uint8_t *zHost;
    uint8_t *xDevice;
    uint8_t *yDevice;
    uint8_t *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, inputX.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, outputByteSize, inputY.data(), outputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ATVC::BroadcastParam param;    // Broadcast运行态参数，包含TilingData以及临时空间的相关信息
    ATVC::BroadcastPolicy policy = {-1, -1, -1};  // Broadcast运行态参数，负责映射最适合的Broadcast模板实现
    // Host侧调用Tiling API完成相关运行态参数的运算
    param.nBufferNum = 1;
    if (!ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param)) {
        printf("Broadcast tiling error.\n");
        return -1;
    };
    // 调用Adapter调度接口，完成核函数的模板调用
    BroadcastOpAdapter<BroadcastOpTraits>(xDevice, yDevice, zDevice, param, policy);

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> outputZ(reinterpret_cast<float*>(zHost), reinterpret_cast<float*>(zHost) + outEleNum);

    CleanUp(xDevice, yDevice, zDevice, zHost);
    CleanACL(stream, context, deviceId);

    if (!VerifyResults(golden, outputZ)) {
        return -1;
    }
    printf("Accuracy verification passed.\n");
    return 0;
}
