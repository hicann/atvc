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
#include "example_common.h"
#include "reduce/host/reduce_host.h"
#include "reduce/kernel/reduce_kernel.h"
#include "relu_with_reduce_sum.h"

namespace {
// ReLUWithReduceSum op description：a input, a output, a temp
using ReduceOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>, ATVC::OpTemps<float>>;

template <class OpTraits>
void ReduceOpAdapter(
    uint8_t *x, uint8_t *y, ATVC::ReduceParam param, ATVC::ReducePolicy &policy, aclrtStream &stream)
{
    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, param.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    param.workspaceAddr = reinterpret_cast<uint64_t>(workspaceDevice);
    if (policy == ATVC::REDUCE_POLICY0) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY0><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY1) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY1><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY2) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY2><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY3) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY3><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY4) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY4><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY5) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY5><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY6) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY6><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY7) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY7><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY8) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY8><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY9) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY9><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY10) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY10><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY11) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY11><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY12) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY12><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY13) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY13><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY14) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY14><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY15) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY15><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY16) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY16><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY17) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY17><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY18) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY18><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY19) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY19><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY20) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY20><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY21) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY21><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY22) {
        ReluWithReduceSum<OpTraits, ATVC::REDUCE_POLICY22><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    }
    // sync stream
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtFree(workspaceDevice));
}

void initData(std::vector<float> &x, std::vector<float> &golden, int eleNum, int outEleNum)
{
    for (int i = 0; i < eleNum; ++i) {
        if (i % 2 == 0) {  // 2 : even number
            x[i] = 2;      // 2 : input data
        } else {
            x[i] = -2;  // -2 : input data
        }
    }
    for (int i = 0; i < outEleNum; ++i) {
        if (i % 2 == 0) {    // 2 : even number
            golden[i] = 24;  // 24 : ret-val
        } else {
            golden[i] = 0;
        }
    }
}

void CleanUp(uint8_t *&xDevice, uint8_t *&yDevice, uint8_t *&yHost)
{
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(yHost));
}
}  // namespace

int32_t main(int32_t argc, char *argv[])
{
    if (!ATVC::Host::DebugCheck<ReduceOpTraits, ATVC::TemplateType::REDUCE>()) {
        printf("[ERROR]: Reduce OpTraits check failed.\n");
        return -1;
    }
    int32_t eleNum = 8 * 1024;
    int32_t outEleNum = 1 * 1024;
    std::vector<int64_t> shapeIn{8, 1024};
    std::vector<int64_t> shapeOut{1, 1024};

    size_t inputByteSize = static_cast<size_t>(eleNum) * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(outEleNum) * sizeof(float);
    std::vector<int64_t> dim{0};
    std::vector<float> x(eleNum);
    std::vector<float> golden(outEleNum);
    initData(x, golden, eleNum,outEleNum); 
    // init Acl
    aclrtContext context;
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    InitializeACL(context,stream,deviceId);
    uint8_t *yHost;
    uint8_t *xDevice;
    uint8_t *yDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&yHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ATVC::ReduceParam param;
    ATVC::ReducePolicy policy = {-1, -1, -1};
    if (!ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shapeIn, dim, &policy, &param)) {
        printf("Reduce tiling error.\n");
        return -1;
    };
    ReduceOpAdapter<ReduceOpTraits>(xDevice, yDevice, param, policy, stream);
#if ATVC_DEBUG_MODE == 2                // 2: open profiling
    for (int32_t i = 0; i < 19; i++) {  // 19: run kernel 1 + 19 times for profiling
        ReduceOpAdapter<ReduceOpTraits>(xDevice, yDevice, param, policy, stream);
    }
#endif
    CHECK_ACL(aclrtMemcpy(yHost, outputByteSize, yDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> y(reinterpret_cast<float *>(yHost), reinterpret_cast<float *>(yHost) + outEleNum);

    // destory Acl
    CleanUp(xDevice, yDevice, yHost);
    CleanACL(stream, context, deviceId);

    if (!VerifyResults(golden, y)) {
        return -1;
    }
    printf("Accuracy verification passed.\n");
    return 0;
}