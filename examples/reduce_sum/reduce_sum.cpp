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
#include "reduce/host/reduce_host.h"
#include "reduce/kernel/reduce_kernel.h"
#include "example_common.h"

namespace {
// ReduceSum算子的描述：一个输入，一个输出，类型均为float
using ReduceOpTraits =  ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;

void CleanUp(uint8_t *&xDevice, uint8_t *&yDevice, uint8_t *&yHost)
{
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(yHost));
}
}

/*
 * 该函数为ReduceCustom算子核函数入口
 * x                 Device上的gm地址，指向Add算子第一个输入
 * y                 Device上的gm地址，指向Add算子第一个输出
 * reduceParam       指向运行态ATVC::ReduceParam数据
*/
template<typename Traits, const auto& Policy>
__global__ __aicore__ void ReduceCustom(GM_ADDR x, GM_ADDR y, ATVC::ReduceParam reduceParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0); // 使用了多核控制指令，设置算子执行时只启动Vector核
    // 将计算模板类模板定义作为模板参数传入，Policy由Host层的策略分派API给出
    auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<Traits>, Policy>();
    op.Run(x, y, &reduceParam);
}
namespace {
// 负责Reduce类算子的调度，选择对应的Policy最佳策略并执行Kernel函数
template<class OpTraits>
void ReduceOpAdapter(uint8_t* x, uint8_t* y, ATVC::ReduceParam &param, ATVC::ReducePolicy &policy, aclrtStream& stream)
{
    // 申请临时空间workspace
    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, param.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    param.workspaceAddr = reinterpret_cast<uint64_t>(workspaceDevice);
    // 将tiling api计算出的ReducePolicy转化为编译态参数并实例化相应的核函数
    if (policy == ATVC::REDUCE_POLICY0) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY0><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY1) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY1><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY2) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY2><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY3) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY3><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY4) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY4><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY5) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY5><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY6) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY6><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY7) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY7><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY8) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY8><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY9) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY9><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY10) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY10><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY11) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY11><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY12) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY12><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY13) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY13><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY14) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY14><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY15) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY15><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY16) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY16><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY17) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY17><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY18) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY18><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY19) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY19><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY20) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY20><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY21) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY21><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY22) {
        ReduceCustom<OpTraits, ATVC::REDUCE_POLICY22><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
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
    if (!ATVC::Host::DebugCheck<ReduceOpTraits, ATVC::TemplateType::REDUCE>()) {
        printf("[ERROR]: Reduce OpTraits check failed.\n");
        return -1;
    }
    int32_t eleNum = 8 * 1024;
    int32_t outEleNum = 1 * 1024;
    size_t inputByteSize = static_cast<size_t>(eleNum) * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(outEleNum) * sizeof(float);
    std::vector<int64_t> dim{0};            // 对第0轴执行reduce操作
    std::vector<int64_t> shape{8, 1024};    // 测试输入shape
    std::vector<float> inputX(eleNum, 1.0f);
    std::vector<float> golden(outEleNum, 8.0f);
    // 初始化Acl资源
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    uint8_t *yHost;
    uint8_t *xDevice;
    uint8_t *yDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&yHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, inputX.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ATVC::ReduceParam param;    // Reduce运行态参数，包含TilingData以及临时空间的相关信息
    ATVC::ReducePolicy policy = {-1, -1, -1};  // Reduce运行态参数，负责映射最适合的Reduce模板实现
    ATVC::Host::ReduceTilingHyperParam hyperParam;
    hyperParam.maxInnerA = 256;// 设置maxInnerA为256
    // Host侧调用Tiling API完成相关运行态参数的运算
    if (!ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param, hyperParam=hyperParam)) {
        printf("Reduce tiling error.\n");
        return -1;
    };

    // 调用Adapter调度接口，完成核函数的模板调用
    ReduceOpAdapter<ReduceOpTraits>(xDevice, yDevice, param, policy, stream);

    CHECK_ACL(aclrtMemcpy(yHost, outputByteSize, yDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> outputY(reinterpret_cast<float*>(yHost), reinterpret_cast<float*>(yHost) + outEleNum);

    // 释放Acl资源
    CleanUp(xDevice, yDevice, yHost);
    CleanACL(stream, context, deviceId);

    if (!VerifyResults(golden, outputY)) {
        return -1;
    }

    printf("Accuracy verification passed.\n");
    return 0;
}
