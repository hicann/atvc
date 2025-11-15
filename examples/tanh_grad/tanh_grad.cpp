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
struct MemoryPtrs {
    uint8_t *zHost;
    uint8_t *dyDevice;
    uint8_t *yDevice;
    uint8_t *zDevice;
    uint8_t *paramDevice;
};

void InitializeData(int32_t eleNum, std::vector<float> &inputDy, std::vector<float> &inputY, std::vector<float> &golden)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (int i = 0; i < eleNum; ++i) {
        inputDy[i] = (dis(gen));
        inputY[i] = (dis(gen));
    }
    for (int i = 0; i < eleNum; ++i) {
        // dy * (1 - x ^ 2)
        golden[i] = (inputDy[i]) * (1 - inputY[i] * inputY[i]);
    }
}

void CleanUp(uint8_t *&zHost, uint8_t *&dyDevice, uint8_t *&yDevice, uint8_t *&zDevice)
{
    CHECK_ACL(aclrtFree(dyDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(zHost));
}

void MallocHostDeviceMemory(MemoryPtrs& memoryPtrs, size_t byteSize, std::vector<float>& inputDy, std::vector<float>& inputY)
{
    CHECK_ACL(aclrtMallocHost((void **)(&memoryPtrs.zHost), byteSize));
    CHECK_ACL(aclrtMalloc((void **)&memoryPtrs.dyDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&memoryPtrs.yDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&memoryPtrs.zDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(memoryPtrs.dyDevice, byteSize, inputDy.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(memoryPtrs.yDevice, byteSize, inputY.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
}

// Add算子中有两个输入，一个输出。类型均为float
using AddOpTraits = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;

// 传入编译态参数ATVC::OpTraits
template <typename Traits>
struct TanhGradComputeFunc {
    /**
    * \brief: Compute operator of tanh: z = dy * (1 - y ^ 2)
    * \param [in] dy, input local tensor
    * \param [in] y, input local tensor
    * \param [out] z, output local tensor
    * \return void
    */
    template <typename DTYPE_DY, typename DTYPE_X, typename DTYPE_Z>
    __aicore__ inline void operator()(
        AscendC::LocalTensor<DTYPE_DY> dy, AscendC::LocalTensor<DTYPE_X> y, AscendC::LocalTensor<DTYPE_Z> z)
    {
        auto length = y.GetSize();
        AscendC::Mul(y, y, y, length);
        AscendC::Mul(y, dy, y, length);
        AscendC::Sub(z, dy, y, length);
    }
};
}

template <class Traits>
/**
* \brief: Kernel entry of Tanh, kernel func is: z = dy * (1 - y ^ 2)
* \param [in] dy, input global tensor
* \param [in] y, input global tensor
* \param [out] z, output global tensor
* \return void
*/
__global__ __aicore__ void TanhGrad(GM_ADDR dy, GM_ADDR y, GM_ADDR z, ATVC::EleWiseParam param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto op = ATVC::Kernel::EleWiseOpTemplate<
        TanhGradComputeFunc<Traits>>();  // 将TanhGradComputeFunc仿函数作为模板参数传入，实例化EleWiseOpTemplate模板类
    op.Run(dy, y, z, &param);  // 按照输入、输出、param的顺序传入Run函数，实现GM->GM的数据计算
}

int main()
{
    if (!ATVC::Host::DebugCheck<AddOpTraits, ATVC::TemplateType::ELE_WISE>()) {
        printf("[ERROR]: ElementWise OpTraits check failed.\n");
        return -1;
    }
    // totalCnt描述EleWise单输入的元素个数
    int32_t eleNum = 8 * 1024;
    size_t byteSize = static_cast<size_t>(eleNum) * sizeof(float);

    std::vector<float> inputDy(eleNum);
    std::vector<float> inputY(eleNum);
    std::vector<float> golden(eleNum);

    // 生成输入数据
    InitializeData(eleNum, inputDy, inputY, golden);
    // 声明运行态参数param
    ATVC::EleWiseParam param;
    ATVC::Host::EleWiseTilingHyperParam hyperParam;
    hyperParam.singleCoreBaseLine = 1024; // set base count for single core 为1024.
    if (!ATVC::Host::CalcEleWiseTiling<AddOpTraits>(eleNum, param, hyperParam=hyperParam)) {
        printf("[ERROR]: Calculate eleWise tiling failed.\n");
        return -1;
    };
    aclrtContext context;
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    InitializeACL(context, stream, deviceId);
    MemoryPtrs memoryPtrs;
    MallocHostDeviceMemory(memoryPtrs, byteSize, inputDy, inputY);
    uint32_t blockNum = param.tilingData.blockNum;
    TanhGrad<AddOpTraits><<<blockNum, nullptr, stream>>>(memoryPtrs.dyDevice, memoryPtrs.yDevice, memoryPtrs.zDevice, param);
#if ATVC_DEBUG_MODE == 2                // 2: open profiling
    for (int32_t i = 0; i < 19; i++) {  // 19: run kernel 1 + 19 times for profiling
        TanhGrad<AddOpTraits><<<blockNum, nullptr, stream>>>(memoryPtrs.dyDevice, memoryPtrs.yDevice, memoryPtrs.zDevice, param);
    }
#endif
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(memoryPtrs.zHost, byteSize, memoryPtrs.zDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> outputZ(reinterpret_cast<float *>(memoryPtrs.zHost), reinterpret_cast<float *>(memoryPtrs.zHost) + eleNum);

    CleanUp(memoryPtrs.zHost, memoryPtrs.dyDevice, memoryPtrs.yDevice, memoryPtrs.zDevice);
    CleanACL(stream, context, deviceId);

    if (!VerifyResults(golden, outputZ)) {
        return -1;
    }
    printf("Accuracy verification passed.\n");
    return 0;
}
