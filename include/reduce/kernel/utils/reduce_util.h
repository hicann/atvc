/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_util.h
 * \brief reduce template struct
 */

#ifndef ATVC_REDUCE_REDUCE_UTIL_H
#define ATVC_REDUCE_REDUCE_UTIL_H
#include "common/const_def.h"
#include "common/kernel_check_debug.h"
#include "reduce/common/patterns.h"

namespace ATVC {
namespace KernelUtils {
template <int32_t dim>
struct Shape {
    int64_t value[dim];
    int64_t oriBurstLen;
};

template <typename T>
__aicore__ inline constexpr int32_t GetCopyInCount()
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value || AscendC::IsSameType<T, half>::value) {
        return CONST2;
    } else {
        return CONST3;
    }
}

template <typename T>
__aicore__ inline constexpr int32_t GetComputeCount()
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value || AscendC::IsSameType<T, half>::value) {
        return CONST2;
    } else {
        return CONST0;
    }
}

template <typename T, const auto& SelectReducePolicy>
__aicore__ inline void PrintReduceParam(const T* param)
{
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: factorACntPerCore = %lu\n",
        param->tilingData.factorACntPerCore);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: factorATotalCnt = %lu\n",
        param->tilingData.factorATotalCnt);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: ubFactorA = %lu\n",
        param->tilingData.ubFactorA);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: factorRCntPerCore = %lu\n",
        param->tilingData.factorRCntPerCore);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: factorRTotalCnt = %lu\n",
        param->tilingData.factorRTotalCnt);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: ubFactorR = %lu\n", param->tilingData.ubFactorR);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: groupR = %lu\n", param->tilingData.groupR);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: outSize = %lu\n", param->tilingData.outSize);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: basicBlock = %lu\n", param->tilingData.basicBlock);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: coreNum = %d\n", param->tilingData.coreNum);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Tiling data: nBufferNum = %d\n", param->nBufferNum);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Work space size is %u\n", param->workspaceSize);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] policy = (%d, %d, %d)\n",
        SelectReducePolicy.patternID, SelectReducePolicy.loopARCount, SelectReducePolicy.loopInnerARCount);
    return;
}

template <typename T>
__aicore__ inline bool CheckReduceParam(const T* param)
{
    auto *tilingData = &param->tilingData;
    if (tilingData->coreNum < AscendC::GetBlockIdx() + 1 ) {
        ATVC::Kernel::DebugPrintf("[ERROR]: [ATVC][Reduce] Tiling data[coreNum = %d] is invalid, it must be larger than current block number.\n", tilingData->coreNum);
        return false;
    }
    if (tilingData->basicBlock % ATVC::UB_ALIGN_32 != 0) {
        ATVC::Kernel::DebugPrintf("[ERROR]: [ATVC][Reduce] Tiling data[basicBlock = %lu] is invalid, it must be divisible by 32.\n", tilingData->coreNum);
        return false;
    }
    return true;
}

} // namespace KernelUtils
} // namespace ATVC
#endif