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
#include "addcmul.h"

namespace {
// AddWithBroadcast算子的描述：两个输入，一个输出，类型均为float
using BroadcastOpTraits = ATVC::OpTraits<ATVC::OpInputs<float, float, float>, ATVC::OpOutputs<float>, ATVC::OpTemps<float, float, float>>;

// 负责Broadcast类算子的调度，选择对应的Policy最佳策略并执行Kernel函数
template<class OpTraits>
void BroadcastOpAdapter(uint8_t* tensor1, uint8_t* tensor2, float value, uint8_t* input, uint8_t* output, ATVC::BroadcastParam &param, ATVC::BroadcastPolicy &policy, aclrtStream& stream)
{
    // 申请临时空间workspace，并将其与BroadcastTilingData一同传到Device侧
    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, param.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    param.workspaceAddr = reinterpret_cast<uint64_t>(workspaceDevice);
    // 将tiling api计算出的BroadcastPolicy转化为编译态参数并实例化相应的核函数
    if (policy == ATVC::BROADCAST_POLICY0) {
        AddcmulCustom<OpTraits, ATVC::BROADCAST_POLICY0><<<param.tilingData.coreNum, nullptr, stream>>>(tensor1, tensor2, input, output, param, value);
     }else if (policy == ATVC::BROADCAST_POLICY1) {
        AddcmulCustom<OpTraits, ATVC::BROADCAST_POLICY1><<<param.tilingData.coreNum, nullptr, stream>>>(tensor1, tensor2, input, output, param, value);
    } else {
        printf("[ERROR] Cannot find any matched policy.\n");
    }
    // 流同步后释放申请的param内存
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtFree(workspaceDevice));
}

void CleanUp(uint8_t *&tensor1Device, uint8_t *&tensor2Device, uint8_t *&inputDevice, uint8_t *&outputDevice,
    uint8_t *&outputHost)
{
    CHECK_ACL(aclrtFree(tensor1Device));
    CHECK_ACL(aclrtFree(tensor2Device));
    CHECK_ACL(aclrtFree(inputDevice));
    CHECK_ACL(aclrtFree(outputDevice));
    CHECK_ACL(aclrtFreeHost(outputHost));
}

void InitializeData(int32_t eleNum, int32_t outEleNum, float value, std::vector<float> &tensor1,
    std::vector<float> &tensor2, std::vector<float> &input, std::vector<float> &golden)
{
    if (eleNum == 0) {
        return;
    }
    std::random_device rd;
    std::mt19937 gen(static_cast<std::mt19937::result_type>(rd()));
    std::uniform_real_distribution<float> disX(1.0f, 9.0f);
    std::uniform_real_distribution<float> disY(1.0f, 9.0f);

    for (int i = 0; i < eleNum; ++i) {
        tensor1[i] = (disX(gen));
        tensor2[i] = (disX(gen));
    }
    for (int i = 0; i < outEleNum; ++i) {
        input[i] = (disY(gen));
    }
    for (int i = 0; i < outEleNum; ++i) {
        golden[i] = input[i] + (tensor1[i % eleNum] * tensor2[i % eleNum] * value);
    }
}
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t eleNum = 1 * 8;
    int32_t outEleNum = 8 * 8;
    std::vector<int64_t> shapeIn{1, 8};    // 测试输入shape
    std::vector<int64_t> shapeOut{8, 8};    // 测试输入shape

    size_t inputByteSize = static_cast<size_t>(eleNum) * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(outEleNum) * sizeof(float);
    std::vector<float> tensor1(eleNum);
    std::vector<float> tensor2(eleNum);
    float value = 4;
    std::vector<float> input(outEleNum);
    std::vector<float> golden(outEleNum);
    InitializeData(eleNum, outEleNum, value, tensor1, tensor2, input, golden);

    // 初始化Acl资源
    aclrtContext context;
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    InitializeACL(context, stream, deviceId);
    uint8_t *outputHost;
    uint8_t *tensor1Device;
    uint8_t *tensor2Device;
    uint8_t *inputDevice;
    uint8_t *outputDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&outputHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&tensor1Device, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tensor2Device, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&inputDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outputDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(tensor1Device, inputByteSize, tensor1.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tensor2Device, inputByteSize, tensor2.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(inputDevice, outputByteSize, input.data(), outputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ATVC::BroadcastParam param;    // Broadcast运行态参数，包含TilingData以及临时空间的相关信息
    ATVC::BroadcastPolicy policy = {-1, -1, -1};  // Broadcast运行态参数，负责映射最适合的Broadcast模板实现
    // Host侧调用Tiling API完成相关运行态参数的运算
    param.nBufferNum = 1;
    if (!ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param)) {
        printf("Broadcast tiling error.\n");
        return -1;
    };
    // 调用Adapter调度接口，完成核函数的模板调用
    BroadcastOpAdapter<BroadcastOpTraits>(tensor1Device, tensor2Device, value, inputDevice, outputDevice, param, policy, stream);

    CHECK_ACL(aclrtMemcpy(outputHost, outputByteSize, outputDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> output(reinterpret_cast<float*>(outputHost), reinterpret_cast<float*>(outputHost) + outEleNum);

    CleanUp(tensor1Device, tensor2Device, inputDevice, outputDevice, outputHost);
    CleanACL(stream, context, deviceId);

    if (!VerifyResults(golden, output)) {
        return -1;
    }
    printf("Accuracy verification passed.\n");
    return 0;
}
