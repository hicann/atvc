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
 * \file reduce_block_aux_util.h
 * \brief reduce schedule dispatch&impl
 */

#ifndef ATVC_REDUCE_BLOCK_AUX_UTIL_H
#define ATVC_REDUCE_BLOCK_AUX_UTIL_H
#include "reduce_util.h"
#include "kernel_operator.h"
#include "common/platform.h"
#include "common/ops_utils_device.h"
#include "reduce/common/patterns.h"
#include "reduce/common/reduce_common.h"

namespace ATVC {
namespace KernelUtils {
namespace Reduce {
constexpr uint16_t MAX_OFFSET = 16;
__aicore__ inline int16_t MainR(int64_t dimR, bool isAR)
{
    if (isAR && dimR < UB_ALIGN_32) {
        return 0;
    }
    int64_t mainR = 1;
    for (uint16_t i = 1; i < MAX_OFFSET; i++) {
        if ((dimR >> i) == 0) {
            break;
        }
        mainR = 1LL << i;
    }
    return static_cast<int16_t>(mainR);
}

__aicore__ inline int64_t GetCacheID(const int64_t idx)
{
    return AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + CONST1)) - CONST1;
}

template <auto LoopInfo>
__aicore__ inline constexpr bool IsBlockCutA() {
    return LoopInfo->loopRCount <= 0;
}

template <auto LoopInfo>
__aicore__ inline bool IsLoopAxis(int32_t axis) {
    if constexpr (LoopInfo->loopACount > 0) {
        for (int32_t i = 0; i < LoopInfo->loopACount; i++) {
            if (LoopInfo->loopAAxis[i] == axis) {
                return true;
            }
        }
    }

    if constexpr (LoopInfo->loopRCount > 0) {
        for (int32_t i = 0; i < LoopInfo->loopRCount; i++) {
            if (LoopInfo->loopRAxis[i] == axis) {
                return true;
            }
        }
    }

    if constexpr (LoopInfo->loopInnerACount > 0) {
        for (int32_t i = 0; i < LoopInfo->loopInnerACount; i++) {
            if (LoopInfo->loopInnerAAxis[i] == axis) {
                return true;
            }
        }
    }

    if constexpr (LoopInfo->loopInnerRCount > 0) {
        for (int32_t i = 0; i < LoopInfo->loopInnerRCount; i++) {
            if (LoopInfo->loopInnerRAxis[i] == axis) {
                return true;
            }
        }
    }
    return false;
}

template <auto LoopInfo>
__aicore__ inline bool IsLoopSpliteRAxis(int32_t axis) {
    if constexpr (LoopInfo->loopInnerRCount) {
        if (LoopInfo->loopInnerRAxis[LoopInfo->loopInnerRCount - 1] == axis) {
            return true;
        }
    }
    if constexpr (LoopInfo->loopRCount) {
        if (LoopInfo->loopRAxis[LoopInfo->loopRCount - 1] == axis) {
            return true;
        }
    }
    return false;
}

template <auto LoopInfo>
__aicore__ inline bool IsLoopSpliteAAxis(int32_t axis) {
    if constexpr (LoopInfo->loopACount) {
        if (LoopInfo->loopAAxis[LoopInfo->loopACount - 1] == axis) {
            return true;
        }
    }
    return false;
}

template <auto LoopInfo, int32_t dim, class T, class U, int32_t desStride = 1>
__aicore__ inline constexpr uint64_t GetBurstLen(T& iterAddr, U& tiling) {
    
    return iterAddr[dim].stride * tiling->stride[dim];
}

template <auto LoopInfo, class T, class U>
__aicore__ inline uint64_t GetRepeatStride(int32_t dim, T& iterAddr, U& tiling, uint64_t& stride) {
    for (int64_t i = dim; i >= 0; i--) {
        if (i == 0) {
            stride = tiling->stride[i];
            return iterAddr[i].stride;
        } else {
            if (iterAddr[i].stride != 1) {
                stride = tiling->stride[i];
                return iterAddr[i].stride;
            }
        }
    }
    return 1;
}

template <auto LoopInfo, class T>
__aicore__ inline uint64_t GetRepeatStrideAxis(int32_t Dim, T& iterAddr) {
    for (int64_t i = Dim; i >= 0; i--) {
        if (i == 0) {
            return i;
        } else {
            if (iterAddr[i].stride != 1) {
                return i;
            }
        }
    }
    return 0;
}

template <bool FirstA>
__aicore__ inline bool IsAxisA(int32_t idx) {
    if constexpr (FirstA) {
        return idx % 2 == 0;
    } else {
        return idx % 2 == 1;
    }
}

template <auto LoopInfo, bool tailA, int32_t dim, class T>
__aicore__ inline int32_t GetInnerA(T& iterAddr) {
    // Traverse from back to front to the innermost dividing axis of A.
    int32_t startAxisA = 0;
    if constexpr (tailA) {
        startAxisA = dim - 1;
    } else {
        startAxisA = dim - 2;
    }

    int32_t endAxisA = -1;
    if constexpr (LoopInfo->loopACount > 0) {
        endAxisA = LoopInfo->loopAAxis[LoopInfo->loopACount - 1];
    }
    int32_t innerA = 1;
    for (int32_t idx = startAxisA; idx > endAxisA; idx -= 2) {
        innerA = innerA * iterAddr[idx].stride;
    }
    return innerA;
}

template <const auto& ReducePolicy>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo0() {
    constexpr ReduceSchLoopInfo schInfo = {.patternID = ReducePolicy.patternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = ReducePolicy.loopARCount / CONST10,
                                           .loopAAxis = {},
                                           .loopRCount = ReducePolicy.loopARCount % CONST10,
                                           .loopRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = ReducePolicy.loopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = ReducePolicy.loopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = ReducePolicy.patternID % CONST10};
    return schInfo;
}

template <const auto& ReducePolicy>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo1() {
    constexpr ReduceSchLoopInfo schInfo = {.patternID = ReducePolicy.patternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = ReducePolicy.loopARCount / CONST10,
                                           .loopAAxis = {DIM0},
                                           .loopRCount = ReducePolicy.loopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = ReducePolicy.loopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = ReducePolicy.loopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = ReducePolicy.patternID % CONST10};
    return schInfo;
}

template <const auto& ReducePolicy>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo2() {
    constexpr ReduceSchLoopInfo schInfo = {.patternID = ReducePolicy.patternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = ReducePolicy.loopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2},
                                           .loopRCount = ReducePolicy.loopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = ReducePolicy.loopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount =ReducePolicy.loopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = ReducePolicy.patternID % CONST10};
    return schInfo;
}

template <const auto& ReducePolicy>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo3() {
    constexpr ReduceSchLoopInfo schInfo = {.patternID = ReducePolicy.patternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = ReducePolicy.loopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2, DIM4},
                                           .loopRCount = ReducePolicy.loopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM4, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = ReducePolicy.loopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = ReducePolicy.loopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = ReducePolicy.patternID % CONST10};
    return schInfo;
}

template <const auto& ReducePolicy>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo4() {
    constexpr ReduceSchLoopInfo schInfo = {.patternID = ReducePolicy.patternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = ReducePolicy.loopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2, DIM4, DIM6},
                                           .loopRCount = ReducePolicy.loopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM4, DIM6, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = ReducePolicy.loopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = ReducePolicy.loopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = ReducePolicy.patternID % CONST10};
    return schInfo;
}

template <const auto& ReducePolicy>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo5() {
    constexpr ReduceSchLoopInfo schInfo = {.patternID = ReducePolicy.patternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = ReducePolicy.loopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopRCount = ReducePolicy.loopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM4, DIM6, DIM8, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = ReducePolicy.loopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = ReducePolicy.loopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = ReducePolicy.patternID % CONST10};
    return schInfo;
}

__aicore__ inline constexpr ReduceSchLoopInfo GetGroupSchLoopInfo()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = ReducePattern::PATTERN_RA,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = CONST1,
                                           .loopAAxis = {DIM1},
                                           .loopRCount = CONST0,
                                           .loopRAxis = {},
                                           .loopInnerACount = CONST0,
                                           .loopInnerAAxis = {},
                                           .loopInnerRCount = CONST1,
                                           .loopInnerRAxis = {DIM0},
                                           .innerPatternID = ReducePattern::PATTERN_RA};
    return schInfo;
}

template <const auto& ReducePolicy>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo() {
    if constexpr (ReducePolicy.loopARCount / CONST10 == CONST0) {
        return GetSchLoopInfo0<ReducePolicy>();
    } else if constexpr (ReducePolicy.loopARCount / CONST10 == CONST1) {
        return GetSchLoopInfo1<ReducePolicy>();
    } else if constexpr (ReducePolicy.loopARCount / CONST10 == CONST2) {
        return GetSchLoopInfo2<ReducePolicy>();
    } else if constexpr (ReducePolicy.loopARCount / CONST10 == CONST3) {
        return GetSchLoopInfo3<ReducePolicy>();
    } else if constexpr (ReducePolicy.loopARCount / CONST10 == CONST4) {
        return GetSchLoopInfo4<ReducePolicy>();
    } else if constexpr (ReducePolicy.loopARCount / CONST10 == CONST5) {
        return GetSchLoopInfo5<ReducePolicy>();
    }
}
}  // namespace Reduce
} // namespace KernelUtils
} // namespace ATVC
#endif