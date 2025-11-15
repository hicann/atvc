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
#include "pool/host/pool_host.h"
#include "pool/kernel/pool_kernel.h"
#include "example_common.h"

namespace {
void InitializeData(ATVC::Layout2Dim totalLayout, std::vector<float> &inputX, std::vector<float> &golden)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);
    for (int h = 0; h < totalLayout.height; ++h) {
        for (int w = 0; w < totalLayout.width; ++w) {
            uint32_t idx = h * totalLayout.width + w;
            inputX[idx] = dis(gen);
        }
    }

    for (int w = 1; w < totalLayout.width - 1; ++w) {
        for (int h = 1; h < totalLayout.height - 1; ++h) {
            uint32_t idx = h * totalLayout.width + w;
            golden[idx] = inputX[idx + 1] + inputX[idx + 1 + totalLayout.width] + inputX[idx + 1 - totalLayout.width];
            golden[idx] -= inputX[idx - 1] + inputX[idx - 1 + totalLayout.width] + inputX[idx - 1 - totalLayout.width];
            golden[idx] = golden[idx] / 3;  // 3: // min(abs((x[1] + x[4] + x[7] / 3), 255)
            golden[idx] = std::abs(golden[idx]);
            golden[idx] = std::min(golden[idx], 255.0f);
        }
    }
}

bool VerifyResults(const std::vector<float> &golden, const std::vector<float> &output, uint32_t totalH, uint32_t totalW)
{
    for (uint32_t w = 1U; w < totalW - 2U; ++w) {      // 2: ignore pading
        for (uint32_t h = 1U; h < totalH - 2U; ++h) {  // 2: ignore pading
            uint32_t i = h * totalW + w;
            if (!IsClose(golden[i], output[i])) {
                printf("[ERROR]: Accuracy verification failed! The expected value of element "
                       "in index [%u] is %f, but actual value is %f.\n",
                    i,
                    golden[i],
                    output[i]);
                return false;
            }
        }
    }
    return true;
}

void CleanUp(uint8_t *&zHost, uint8_t *&xDevice, uint8_t *&zDevice)
{
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(zHost));
}

using PoolOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>, ATVC::OpTemps<int32_t>>;

// 传入编译态参数ATVC::OpTraits
template <typename Traits>
struct Edge2C3ComputeFunc {
    static constexpr ATVC::Layout2Dim TILE_LAYOUT{16, 16};  // 基本块宽高 宽需要32B对齐 未裁剪前的
    static constexpr ATVC::PoolTilePadding TILE_PADDING{
        8, 8, 1, 1};  // tile块上下左右padding的设置left/right需要32B对齐 未裁剪前基础值
    /*
    函数说明： c = a + b
    参数说明：
        a                   : 参与运算的输入
        c                   : 参与运算的输出
    */
    template <typename T, typename U>
    // 重载operator，提供给算子模板类调用
    __aicore__ inline void operator()(
        AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> c, AscendC::LocalTensor<U> temp)
    {
        uint32_t calcSize = c.GetSize();
        uint32_t sizeT = sizeof(T);
        static constexpr uint32_t TENSOR_WIDTH = TILE_PADDING.left + TILE_LAYOUT.width + TILE_PADDING.right;
        // 0 1 2
        // 3 4 5
        // 6 7 8
        // 计算: x[1,4,7]: x[2,5,8] - x[0,3,6]
        AscendC::CreateVecIndex<U>(temp, (int32_t)2, calcSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<U>(temp, temp, sizeT, calcSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<uint32_t> tempRef = temp.template ReinterpretCast<uint32_t>();
        AscendC::Gather(c, a, tempRef, 0, calcSize - 2);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(a, c, a, calcSize);
        AscendC::Adds<U>(temp, temp, (sizeT * -3), calcSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Relu(temp, temp, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(c, a, tempRef, 0, calcSize - 2);
        // 计算: x[4]: min(abs((x[1] + x[4] + x[7] / 3), 255)
        AscendC::Add(a[TENSOR_WIDTH], c, c[TENSOR_WIDTH * 2], calcSize - TENSOR_WIDTH * 2);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(c, a, c, calcSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(c, c, 1 / 3.0f, calcSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(c, c, calcSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(c, c, 255.0f, calcSize);
    }
};
}  // namespace

static constexpr ATVC::Layout2Dim totalLayout{1023, 2517};  // 原图宽高
template <class Traits, const auto &totalLayout>
__global__ __aicore__ void EdgeCustom(GM_ADDR a, GM_ADDR c, ATVC::PoolParam param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 将Edge2C3ComputeFunc仿函数作为模板参数传入，实例化PoolOpTemplate模板类
    auto op = ATVC::Kernel::PoolOpTemplate<Edge2C3ComputeFunc<Traits>, totalLayout>();
    op.Run(a, c, &param);  // 按照输入、输出、param的顺序传入Run函数，实现GM->GM的数据计算
}

int main()
{
    // totalCnt描述Pool单输入的元素个数
    int32_t eleNum = totalLayout.width * totalLayout.height;
    size_t inputByteSize = static_cast<size_t>(eleNum) * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(eleNum) * sizeof(float);

    std::vector<float> inputX(eleNum);
    std::vector<float> golden(eleNum);
    InitializeData(totalLayout, inputX, golden);

    aclrtContext context;
    aclrtStream stream = nullptr;
    int32_t deviceId = 0;
    InitializeACL(context, stream, deviceId);

    ATVC::PoolParam param;
    if (!ATVC::Host::CalcPoolTiling<PoolOpTraits>(totalLayout, Edge2C3ComputeFunc<PoolOpTraits>::TILE_LAYOUT, param)) {
        (void)printf("[ERROR]: Calculate Element wise tiling Failed.\n");
        return -1;
    };
    auto elementParamSize = sizeof(param);

    uint8_t *zHost;
    uint8_t *xDevice;
    uint8_t *zDevice;

    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&xDevice), inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&zDevice), outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, inputX.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    uint32_t blockNum = param.tilingData.blockNum;
    // 调用核函数
    (void)printf("blockNum: %u\n", blockNum);
    EdgeCustom<PoolOpTraits, totalLayout><<<blockNum, nullptr, stream>>>(xDevice, zDevice, param);
#if ATVC_DEBUG_MODE == 2                // 2: open profiling
    for (int32_t i = 0; i < 19; i++) {  // 19: run kernel 1 + 19 times for profiling
        EdgeCustom<PoolOpTraits, totalLayout><<<blockNum, nullptr, stream>>>(xDevice, zDevice, param);
    }
#endif
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> outputZ(reinterpret_cast<float *>(zHost), reinterpret_cast<float *>(zHost) + eleNum);

    CleanUp(zHost, xDevice, zDevice);
    CleanACL(stream, context, deviceId);

    if (!VerifyResults(golden, outputZ, totalLayout.height, totalLayout.width)) {
        return -1;
    }
    (void)printf("Accuracy verification passed.\n");
    return 0;
}