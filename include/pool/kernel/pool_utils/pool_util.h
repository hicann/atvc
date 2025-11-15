/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_POOL_UTIL_H
#define ATVC_POOL_UTIL_H
#include "common/const_def.h"
#include "common/kernel_check_debug.h"

namespace ATVC {
namespace KernelUtils {
template <typename T>
__aicore__ inline void PrintPoolParam(const T* param)
{
    ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] Tiling data: blockNum = %u\n", param->tilingData.blockNum);
    ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] Tiling data: tiledCnt = %u\n", param->tilingData.tiledCnt);
    ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] Tiling data: tailBlockCnt = %u\n", param->tilingData.tailBlockCnt);
    ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] Tiling data: numPerBlock = %u\n", param->tilingData.numPerBlock);
    ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] Tiling data: tailElemCnt = %u\n", param->tilingData.tailElemCnt);
    ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] Param: nBufferNum = %u\n", param->nBufferNum);
    ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] Param: totalCnt = %u\n", param->totalCnt);
    return;
}

template <typename T>
__aicore__ inline bool CheckPoolParam(const T* param)
{
    auto *tilingData = &param->tilingData;
    if (tilingData->blockNum < AscendC::GetBlockIdx() + 1) {
        ATVC::Kernel::DebugPrintf("[ERROR]: [ATVC][Pool] Tiling data[blockNum = %u] is invalid,"
                                  "it must be larger than current block number.\n", tilingData->blockNum);
        return false;
    }
    return true;
}

} // namespace KernelUtils
} // namespace ATVC
#endif