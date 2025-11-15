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
 * \file tensor_info.h
 * \brief
 */
#ifndef ATVC_TENSOR_INFO_H
#define ATVC_TENSOR_INFO_H

#include <type_traits>
#include "kernel_operator.h"
#include "type_list.h"
#include "forward.h"
#include "tuple.h"

namespace ATVC {
/**
 * Record the offset position of a LocalTensor in In/Out/Local, as well as its own type
 */
template <typename T>
struct TensorInfo {
    using DataType = T;
    AscendC::GlobalTensor<T> gmTensor;
    int32_t localOffset;
};

template <typename T>
struct TypeToTensor {
    using Type = TensorInfo<T>;
};

template <typename List>
struct TensorTuple {
private:
    using Tensors = typename ATVC::TypeListMap<List, TypeToTensor>::Type;

public:
    using Type = typename ATVC::TupleFromTypeList<Tensors>::Type;
};

/**
 * Get the LocalTensor corresponding to the Tuple element at a specific index
 */
template <std::size_t N, typename TupleType>
__aicore__ inline auto TupleElemGetLocalTensor(AscendC::LocalTensor<uint8_t> local, TupleType& tuple, uint32_t size)
{
    using DataType = typename ATVC::TupleElemType<N, TupleType>::Type::DataType;
    if constexpr (N == 0) {
        auto tensor = local[tuple.head.localOffset].template ReinterpretCast<DataType>();
        tensor.SetSize(size);
        return tensor;
    } else {
        return TupleElemGetLocalTensor<N - 1>(local, tuple.tail, size);
    }
}

} // namespace ATVC

#endif // ATVC_TENSOR_INFO_H
