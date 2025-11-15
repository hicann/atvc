/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_ADD_WITH_BROADCAST_H
#define ATVC_ADD_WITH_BROADCAST_H
#include "post_compute_add_of_broadcast.h"

/* !
 * \brief z = x + y, the shape of x must be able to be broadcasted to the shape of y
 * \param [in] x, input global memory of x
 * \param [in] y, input global memory of y
 * \param [out] z, output global memory
 * \param [in] broadcastParam, params of broadcast
 */
template<typename Traits, const auto& Policy>
__global__ __aicore__ void AddWithBroadcastCustom(GM_ADDR x,
                                                  GM_ADDR y,
                                                  GM_ADDR z,
                                                  ATVC::BroadcastParam broadcastParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 1. get input and output for kernel op from host Traits
    using KernelOpIn = typename Traits::In::types;
    using KernelOpOut = typename Traits::Out::types;
    using KernelOpTemp = typename Traits::Temp::types;

    // 2. define input and output for broadcast
    using BroadcastOpInput = ATVC::OpInputs<typename ATVC::TypeListGet<KernelOpIn, 0>::Type>;
    using BroadcastOpOutput = ATVC::OpOutputs<typename ATVC::TypeListGet<KernelOpTemp, 0>::Type>;
    using BroadcastOpTraits =  ATVC::OpTraits<BroadcastOpInput, BroadcastOpOutput>;

    // 3. define input and output for post compute
    using AddOpInput = ATVC::OpInputs<typename ATVC::TypeListGet<KernelOpTemp, 0>::Type, typename ATVC::TypeListGet<KernelOpIn, 1>::Type>;
    using AddOpOutput = ATVC::OpOutputs<typename ATVC::TypeListGet<KernelOpOut, 0>::Type>;
    using AddOpTraits = ATVC::OpTraits<AddOpInput, AddOpOutput>;
    using PostCompute = PostComputeAddOfBroadcast<AddOpTraits>;

    // 4. call op run
    auto op = ATVC::Kernel::BroadcastOpTemplate<ATVC::BroadcastCompute<BroadcastOpTraits>, Policy, void, PostCompute>();
    ATVC::BroadcastParam *param = &broadcastParam;
    op.Run(x, y, z, param);
}
#endif
