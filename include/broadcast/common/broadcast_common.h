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
 * \file broadcast_common.h
 * \brief
 */
#ifndef ATVC_BROADCAST_COMMON_H
#define ATVC_BROADCAST_COMMON_H
#include "common/const_def.h"
#include "broadcast/common/patterns.h"

namespace ATVC {
namespace AB_PATTERN {
static constexpr uint32_t A = 100;
static constexpr uint32_t AB = 11;
static constexpr uint32_t ABA = 20;
static constexpr uint32_t ABAB = 31;
static constexpr uint32_t ABABA = 40;
};  // namespace AB_PATTERN

struct BroadcastPolicy {
public:
    int32_t patternID = -1;
    int32_t loopABCount = -1;
    int32_t loopInnerABCount = -1;
    constexpr BroadcastPolicy(int patternID, int loopABCount, int loopInnerABCount):
    patternID(patternID), loopABCount(loopABCount),loopInnerABCount(loopInnerABCount){}
    bool operator==(const BroadcastPolicy& rhs) const
    {
        return this->patternID == rhs.patternID && this->loopABCount == rhs.loopABCount &&
               this->loopInnerABCount == rhs.loopInnerABCount;
    }
};

struct BroadcastOpTilingData {
    uint64_t factorACntPerCore = 1;
    uint64_t factorATotalCnt = 1;
    uint64_t factorBCntPerCore = 1;
    uint64_t factorBTotalCnt = 1;
    uint64_t A0 = 1;
    uint64_t A11 = 1;
    uint64_t A12 = 1;
    uint64_t A2 = 1;
    uint64_t B0 = 1;
    uint64_t B1 = 1;
    uint64_t B2 = 1;
    uint64_t basicBlock = 1;
    int32_t coreNum = 1;
    uint64_t shape[ATVC::MAX_DIM] = {1};
    uint64_t dstShape[ATVC::MAX_DIM] = {1};
    uint64_t stride[ATVC::MAX_DIM];
    uint64_t dstStride[ATVC::MAX_DIM];
};

struct BroadcastParam {
    uint64_t workspaceAddr;
    uint32_t workspaceSize = 0;
    BroadcastOpTilingData tilingData;
    int32_t nBufferNum = 2;
};

static constexpr BroadcastPolicy BROADCAST_POLICY0{ATVC::AB_PATTERN::AB, 10, 1};
static constexpr BroadcastPolicy BROADCAST_POLICY1{ATVC::AB_PATTERN::ABA, 10, 1};
};  // namespace ATVC

#endif  // ATVC_BROADCAST_COMMON_H