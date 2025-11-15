/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_KERNEL_UTILS_H
#define ATVC_KERNEL_UTILS_H

#include "common/const_def.h"
#include "kernel_operator.h"
namespace ATVC {
template <AscendC::HardEvent EVENT>
__aicore__ inline void SetEvent(AscendC::HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<EVENT>(eventId);
    AscendC::WaitFlag<EVENT>(eventId);
}
template <AscendC::HardEvent EVENT>
__aicore__ inline void SyncDataQueue()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
    AscendC::SetFlag<EVENT>(eventId);
    AscendC::WaitFlag<EVENT>(eventId);
}

namespace KernelUtils {
template <typename DType>
struct GetPromoteType {};

template <>
struct GetPromoteType<float> {
    using T = float;
};

template <>
struct GetPromoteType<int32_t> {
    using T = int32_t;
};

template <>
struct GetPromoteType<int64_t> {
    using T = int64_t;
};

template <>
struct GetPromoteType<uint8_t> {
    using T = uint8_t;
};

template <>
struct GetPromoteType<half> {
    using T = float;
};
#if __NPU_ARCH__ == 2201
template <>
struct GetPromoteType<bfloat16_t> {
    using T = float;
};
#endif

__aicore__ inline int64_t FindNearestPower(const int64_t value)
{
    if (value == 0) {
        return 0;
    } else if (value <= CONST2) {
        return 1;
    } else if (value <= CONST4) {
        return CONST2;
    } else {
        const int64_t num = value - 1;
        const int64_t pow = CONST63 - clz(num);
        return (1 << pow);
    }
}

__aicore__ inline int64_t CalLog(int64_t value)
{
    int64_t res = 0;
    while (value > 1) {
        value = value >> 1;
        res++;
    }
    return res;
}
} // namespace KernelUtils
} // namespace ATVC
#endif // ATVC_KERNEL_UTILS_H