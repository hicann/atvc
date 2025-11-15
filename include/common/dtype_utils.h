/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_DTYPE_UTILS_H
#define ATVC_DTYPE_UTILS_H
#include "graph/types.h"
namespace ATVC {
template <typename T>
inline ge::DataType GetOriInputType()
{
    if constexpr (std::is_same<T, float>::value) {
        return ge::DataType::DT_FLOAT;
    } else if constexpr (std::is_same<T, int8_t>::value) {
        return ge::DataType::DT_INT8;
    } else if constexpr (std::is_same<T, int16_t>::value) {
        return ge::DataType::DT_INT16;
    } else if constexpr (std::is_same<T, int32_t>::value) {
        return ge::DataType::DT_INT32;
    } else if constexpr (std::is_same<T, int64_t>::value) {
        return ge::DataType::DT_INT64;
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        return ge::DataType::DT_UINT8;
    } else {
        return ge::DataType::DT_UNDEFINED;
    }
};

inline ge::DataType GetPromoteDataType(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DataType::DT_BF16:
        case ge::DataType::DT_FLOAT16:
        case ge::DataType::DT_FLOAT:
            return ge::DataType::DT_FLOAT;
        case ge::DataType::DT_INT8:
            return ge::DataType::DT_INT8;
        case ge::DataType::DT_INT16:
            return ge::DataType::DT_INT16;
        case ge::DataType::DT_INT32:
            return ge::DataType::DT_INT32;
        case ge::DataType::DT_INT64:
            return ge::DataType::DT_INT64;
        case ge::DataType::DT_UINT8:
            return ge::DataType::DT_UINT8;
        case ge::DataType::DT_UINT16:
            return ge::DataType::DT_UINT16;
        case ge::DataType::DT_UINT32:
            return ge::DataType::DT_UINT32;
        case ge::DataType::DT_UINT64:
            return ge::DataType::DT_UINT64;
        default:
            return ge::DataType::DT_UNDEFINED;
    }
}
} // namespace ATVC

#endif