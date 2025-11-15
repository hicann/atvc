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
#include "elewise/host/elewise_host.h"
#include "elewise/kernel/elewise_kernel.h"
#include "example_common.h"

namespace {
using OP_TRAITS =  ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>, ATVC::OpTemps<float>>;

template<typename Traits>
struct AddComputeFunc {
    template<typename T>
    __aicore__ inline void operator()(AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b,
        AscendC::LocalTensor<T> c, AscendC::LocalTensor<T> temp, bool conditionVal) {
            if (conditionVal) {
                AscendC::Add(temp, a, a, c.GetSize()); // temp = a + a
                AscendC::Add(c, temp, b, c.GetSize()); // c = temp + b
            } else {
                AscendC::Add(temp, a, a, c.GetSize()); // temp = a + a
                AscendC::Sub(c, temp, b, c.GetSize()); // c = temp - b
            }
    }
};

void InitializeData(int32_t eleNum, std::vector<float> &inputX, std::vector<float> &inputY, std::vector<float> &golden,
    bool conditionVal)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (int i = 0; i < eleNum; ++i) {
            inputX[i] = dis(gen);
            inputY[i] = dis(gen);
            if (conditionVal) {
                golden[i] = 2 * (inputX[i]) + (inputY[i]);  // z = 2 * x + y
            } else {
                golden[i] = 2 * (inputX[i]) - (inputY[i]);  // z = 2 * x - y
            }
    }
}

void CleanUp(uint8_t *&zHost, uint8_t *&xDevice, uint8_t *&yDevice, uint8_t *&zDevice)
{
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(zHost));
}
}

/*
 * 该函数为AddCustom算子核函数入口
 * a                 Device上的gm地址，指向算子第一个输入
 * b                 Device上的gm地址，指向算子第二个输入
 * c                 Device上的gm地址，指向算子第一个输出
 * param             指向运行态ATVC::EleWiseParam数据
 * conditionVal      标量，控制算子的计算逻辑
*/
template<class Traits>
__global__ __aicore__ void AddCustom(GM_ADDR a, GM_ADDR b, GM_ADDR c, ATVC::EleWiseParam param, bool conditionVal)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto op = ATVC::Kernel::EleWiseOpTemplate<AddComputeFunc<Traits>>(); // 传入计算仿函数<In, Out, Temp>, 实例化算子
    op.Run(a, b, c, &param, conditionVal);                                // 调用Run函数, 执行算子
}


int main()
{
    int32_t eleNum = 8 * 1024;
    size_t inputByteSize = static_cast<size_t>(eleNum) * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(eleNum) * sizeof(float);

    std::vector<float> inputX(eleNum);
    std::vector<float> inputY(eleNum);
    std::vector<float> golden(eleNum);
    bool conditionVal = false;
    // 生成输入数据
    InitializeData(eleNum, inputX, inputY, golden, conditionVal);

    aclrtContext context;
    aclrtStream stream = nullptr;
    int32_t deviceId = 0;
    InitializeACL(context, stream, deviceId);

    ATVC::EleWiseParam param;
    if (!ATVC::Host::CalcEleWiseTiling<OP_TRAITS>(eleNum, param)) {
            printf("Elewise tiling error.\n");
            return -1;
    };
    auto elementParamSize = sizeof(param);

    uint8_t *zHost;
    uint8_t *xDevice;
    uint8_t *yDevice;
    uint8_t *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)&zHost, outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, inputX.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, inputY.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    AddCustom<OP_TRAITS>
        <<<param.tilingData.blockNum, nullptr, stream>>>(xDevice, yDevice, zDevice, param, conditionVal);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> outputZ(reinterpret_cast<float *>(zHost), reinterpret_cast<float *>(zHost) + eleNum);

    CleanUp(zHost, xDevice, yDevice, zDevice);
    CleanACL(stream, context, deviceId);

    if (!VerifyResults(golden, outputZ)) {
            return -1;
    }
    printf("Accuracy verification passed.\n");
    return 0;
}