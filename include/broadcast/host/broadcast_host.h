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
 * \file broadcast_host.h
 * \brief
 */
#ifndef ATVC_BROADCAST_HOST_H
#define ATVC_BROADCAST_HOST_H
#include <vector>
#include "common/atvc_opdef.h"
#include "common/atvc_op_check.h"
#include "common/const_def.h"
#include "broadcast/common/broadcast_common.h"
#include "broadcast/host/tiling/broadcast_tiling.h"

namespace ATVC {
namespace Host {
/*!
 * \brief Generates tiling parameters and policy for the Broadcast Template.
 * \param[in] shapeIn, source tensor shape (may be broadcast to match `shapeOut`).
 * \param[in] shapeOut, destination tensor shape after broadcasting.
 * \param[out] policy, static policy of Broadcast Template.
 * \param[out] param, dynamic param of Broadcast Template.
 * \return true – successfully, false – error.
 */
template<class OpTraits, class PreComputeTraits = void, class PostComputeTraits = void>
bool CalcBroadcastTiling(std::vector<int64_t> shapeIn,
                         std::vector<int64_t> shapeOut,
                         BroadcastPolicy* policy,
                         BroadcastParam* param)
{
    if (policy == nullptr || param == nullptr) {
        printf("[ERROR]: [ATVC][Broadcast] Invalid input: policy or param is null pointer!\n");
        return false;
    }

    using inputDTypeList = typename OpTraits::In::types;
    using DataType = typename ATVC::TypeListGet<inputDTypeList, 0>::Type;
    auto inputDtype = GetOriInputType<DataType>();
    BroadcastTilingInputParam opInput = {shapeIn, shapeOut, inputDtype};
    OpTiling::BroadcastOpTiling<OpTraits, PreComputeTraits, PostComputeTraits> tiling(opInput, policy, param);
    if (!tiling.Run()) {
        printf("[ERROR]: [ATVC][Broadcast] Run tiling failed!\n");
        return false;
    }
    return true;
};
} // Host
} // ATVC
#endif // ATVC_BROADCAST_HOST_H