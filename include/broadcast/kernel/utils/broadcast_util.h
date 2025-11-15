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
 * \file broadcast_util.h
 * \brief broadcast util interface
 */

#ifndef ATVC_BROADCAST_UTIL_H
#define ATVC_BROADCAST_UTIL_H
#include "common/const_def.h"
#include "common/kernel_check_debug.h"

namespace ATVC {
namespace KernelUtils {
template <typename T, const auto& SelectBroadcastPolicy>
__aicore__ inline void PrintBroadcastParam(const T* param)
{
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: A0 = %lu\n", param->tilingData.A0);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: A11 = %lu\n", param->tilingData.A11);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: A12 = %lu\n", param->tilingData.A12);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: A2 = %lu\n", param->tilingData.A2);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: B0 = %lu\n", param->tilingData.B0);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: B1 = %lu\n", param->tilingData.B1);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: B2 = %lu\n", param->tilingData.B2);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: coreNum = %d\n", param->tilingData.coreNum);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: basicBlock = %lu\n", param->tilingData.basicBlock);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: factorACntPerCore = %lu\n",
                              param->tilingData.factorACntPerCore);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: factorATotalCnt = %lu\n",
                              param->tilingData.factorATotalCnt);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: factorBCntPerCore = %lu\n",
                              param->tilingData.factorBCntPerCore);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: factorBTotalCnt = %lu\n",
                              param->tilingData.factorBTotalCnt);
    for (int32_t i = 0; i < ATVC::CONST2; i++) {
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: shape[%d] = %lu\n",
                                  i, param->tilingData.shape[i]);
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: dstShape[%d] = %lu\n",
                                  i, param->tilingData.dstShape[i]);
    }
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: policy.patternID = %d\n",
                              SelectBroadcastPolicy.patternID);
    ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Tiling data: workspaceSize = %u\n", param->workspaceSize);
    return;
}

template <typename T>
__aicore__ inline bool CheckBroadcastParam(const T* param)
{
    auto *tilingData = &param->tilingData;
    if (tilingData->coreNum < AscendC::GetBlockIdx() + 1 ) {
        ATVC::Kernel::DebugPrintf("[ERROR]: [ATVC][Broadcast] Tiling data[coreNum = %d] is invalid,"
                                  "it must be larger than current block number.\n", tilingData->coreNum);
        return false;
    }
    return true;
}

} // namespace KernelUtils
} // namespace ATVC
#endif