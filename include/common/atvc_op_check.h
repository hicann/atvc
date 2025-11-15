/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_COMMON_OP_CHECK_H
#define ATVC_COMMON_OP_CHECK_H

#include <stdio.h>
#include "atvc_opdef.h"

namespace ATVC {
namespace Host {

template <typename InputTypeList, typename OutputTypeList>
bool CheckSameDtype_() {
    if (GetOriInputType<typename ATVC::TypeListGet<InputTypeList, 0>::Type>() !=
        GetOriInputType<typename ATVC::TypeListGet<OutputTypeList, 0>::Type>()) {
        return false;
    }
    return true;
}
template <typename OpTraits, ATVC::TemplateType templateType>
// 校验Traits
bool DebugCheck() {
    using Inputs = typename OpTraits::In::types;
    using Outputs = typename OpTraits::Out::types;
    if constexpr (templateType == ATVC::TemplateType::ELE_WISE) {
        if constexpr (TypeListSize<Inputs>::VALUE == 0) { // 0: input is empty
            printf("[ERROR]: [ATVC][OpTraits] Input can not be empty in Ele-wise template.\n");
            return false;
        }
        if constexpr (TypeListSize<Outputs>::VALUE == 0) { // 0: output is empty
            printf("[ERROR]: [ATVC][OpTraits] Output can not be empty in Ele-wise template.\n");
            return false;
        }
    } else if constexpr (templateType == ATVC::TemplateType::REDUCE) {
        if constexpr (TypeListSize<Inputs>::VALUE != 1) { // 1: input number must be 1.
            printf("[ERROR]: [ATVC][OpTraits] Input numer must be 1 in Reduce template.\n");
            return false;
        }
        if constexpr (TypeListSize<Outputs>::VALUE != 1) { // 1: output number must be 1.
            printf("[ERROR]: [ATVC][OpTraits] Output numer must be 1 in Reduce template.\n");
            return false;
        }
    } else if constexpr (templateType == ATVC::TemplateType::BROADCAST) {
        if constexpr (TypeListSize<Inputs>::VALUE != 1) { // 1: input number must be 1.
            printf("[ERROR]: [ATVC][OpTraits] Input numer must be 1 in broadcast template.\n");
            return false;
        }
        if constexpr (TypeListSize<Outputs>::VALUE != 1) { // 1: input number must be 1.
            printf("[ERROR]: [ATVC][OpTraits] Output numer must be 1 in broadcast template.\n");
            return false;
        }
    }

    if constexpr (templateType == ATVC::TemplateType::REDUCE ||
                  templateType == ATVC::TemplateType::BROADCAST) {
        if (!CheckSameDtype_<Inputs, Outputs>()) {
            printf("[ERROR]: [ATVC][OpTraits] Different input/output data types is not support in Reduce or Broadcast template.\n");
            return false;
        }
    }
    return true;
}
}
}
#endif

