/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_COMMON_OPDEF_H
#define ATVC_COMMON_OPDEF_H

#include "type_list.h"
#include "common/dtype_utils.h"
namespace ATVC {
enum class ParamType {
    INPUT,
    OUTPUT,
    TEMP,
};

enum class TemplateType {
    ELE_WISE,
    REDUCE,
    BROADCAST,
};

template <ParamType paramType_, typename... Ts>
struct ParamTypes {
    using types = ATVC::TypeList<Ts...>;
    static constexpr ParamType usage = paramType_;
};

template <typename... Ts>
using OpInputs = ParamTypes<ParamType::INPUT, Ts...>;

template <typename... Ts>
using OpOutputs = ParamTypes<ParamType::OUTPUT, Ts...>;

template <typename... Ts>
using OpTemps = ParamTypes<ParamType::TEMP, Ts...>;

template <typename InTypeList, typename OutTypeList, typename TempTypeList = ATVC::OpTemps<>>
struct OpTraits {
    using In = InTypeList;
    using Out = OutTypeList;
    using Temp = TempTypeList;
};

struct VoidComputeTraits {
    using In = OpInputs<>;
    using Out = OpOutputs<>;
    using Temp = OpTemps<>;
};

template<typename TileCompute>
struct GetFunctionTraits {
    using ComputeTraits = VoidComputeTraits;
};

template <template <typename> class TileCompute, typename Traits>
struct GetFunctionTraits<TileCompute<Traits>> {
    using ComputeTraits = Traits;
};
} // namespace ATVC
#endif
