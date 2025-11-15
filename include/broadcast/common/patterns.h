/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_BROADCAST_PATTERNS_H
#define ATVC_BROADCAST_PATTERNS_H
#include "common/const_def.h"

namespace ATVC {
namespace BroadcastPattern {
constexpr int32_t MAX_LOOP_DIM = (ATVC::MAX_DIM + 1) / 2;
constexpr int32_t PATTERN_A = 10;
constexpr int32_t PATTERN_BA = 0;
constexpr int32_t PATTERN_AB = 1;
constexpr int32_t PATTERN_ABA = 2;
constexpr int32_t PATTERN_ABAB = 3;
constexpr int32_t PATTERN_ABABA = 4;
constexpr int32_t PATTERN_ABABAB = 5;
constexpr int32_t PATTERN_ABABABA = 6;
constexpr int32_t PATTERN_ABABABAB = 7;
constexpr int32_t PATTERN_ABABABABA = 8;

template <uint32_t id, bool firstA, bool tailA, int32_t dim>
struct PatternConstInfo {
    constexpr static uint32_t ID = id;
    constexpr static bool FirstA = firstA;
    constexpr static bool TailA = tailA;
    constexpr static int32_t Dim = dim;
};

struct A : public PatternConstInfo<PATTERN_A, true, true, CONST1> {};
struct BA : public PatternConstInfo<PATTERN_BA, false, true, CONST2> {};
struct AB : public PatternConstInfo<PATTERN_AB, true, false, CONST2> {};
struct ABA : public PatternConstInfo<PATTERN_ABA, true, true, CONST3> {};
}  // namespace BroadcastPattern
}  // namespace ATVC
#endif  // ATVC_BROADCAST_PATTERNS_H