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
void InitializeData(int32_t eleNum, std::vector<float> &inputX, std::vector<float> &inputY, std::vector<float> &golden)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (int i = 0; i < eleNum; ++i) {
        inputX[i] = dis(gen);
        inputY[i] = dis(gen);
        golden[i] = inputX[i] + inputY[i];
    }
}

void CleanUp(uint8_t *&zHost, uint8_t *&xDevice, uint8_t *&yDevice, uint8_t *&zDevice)
{
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(zHost));
}

// Add算子中有两个输入，一个输出。类型均为float
using AddOpTraits = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;

// 传入编译态参数ATVC::OpTraits
template<typename Traits>
struct AddComputeFunc {
    /*
    函数说明： c = a + b
    参数说明：
        a                   : 参与运算的输入
        b                   : 参与运算的输入
        c                   : 参与运算的输出
    */
    template<typename T>
    // 重载operator，提供给算子模板类调用
    __aicore__ inline void operator()(AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b, AscendC::LocalTensor<T> c) {
        AscendC::Add(c, a, b, c.GetSize()); // 开发调用AscendC Api自行实现计算逻辑, 通过c.GetSize()获取单次计算的元素数量
    }
};
}

template<class Traits>
/*
 * 该函数为Add算子核函数入口
 * a        Device上的gm地址，指向Add算子第一个输入
 * b        Device上的gm地址，指向Add算子第二个输入
 * c        Device上的gm地址，指向Add算子第一个输出
 * param    指向运行态ATVC::EleWiseParam数据
*/
__global__ __aicore__ void AddCustom(GM_ADDR a, GM_ADDR b, GM_ADDR c, ATVC::EleWiseParam param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 将AddComputeFunc仿函数作为模板参数传入，实例化EleWiseOpTemplate模板类
    auto op = ATVC::Kernel::EleWiseOpTemplate<AddComputeFunc<Traits>>();
    op.Run(a, b, c, &param); // 按照输入、输出、param的顺序传入Run函数，实现GM->GM的数据计算
}

int main()
{
    if (!ATVC::Host::DebugCheck<AddOpTraits, ATVC::TemplateType::ELE_WISE>()) {
        printf("[ERROR]: Element wise OpTraits check failed.\n");
        return -1;
    }
    // totalCnt描述EleWise单输入的元素个数
    int32_t eleNum = 8 * 1024;
    size_t inputByteSize = static_cast<size_t>(eleNum) * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(eleNum) * sizeof(float);

    std::vector<float> inputX(eleNum);
    std::vector<float> inputY(eleNum);
    std::vector<float> golden(eleNum);
    InitializeData(eleNum, inputX, inputY, golden);

    aclrtContext context;
    aclrtStream stream = nullptr;
    int32_t deviceId = 0;
    InitializeACL(context, stream, deviceId);

    ATVC::EleWiseParam param;
    if (!ATVC::Host::CalcEleWiseTiling<AddOpTraits>(eleNum, param)) {
        printf("[ERROR]: Calculate Element wise tiling Failed.\n");
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
    uint32_t blockNum = param.tilingData.blockNum;
    // 调用核函数 
    AddCustom<AddOpTraits><<<blockNum, nullptr, stream>>>(xDevice, yDevice, zDevice, param);

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