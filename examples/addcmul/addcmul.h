/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_ADDCMUL_H
#define ATVC_ADDCMUL_H
#include "pre_compute_mul_of_broadcast.h"
#include "post_compute_add_of_broadcast.h"

/* !
 * \brief Addcmul(i) = input(i) + value * tensor1(i) * tensor2(i)
 * \param [in] tensor1, input global memory of tensor1
 * \param [in] tensor2, input global memory of tensor2
 * \param [in] input, input global memory of input
 * \param [out] output, output global memory
 * \param [in] broadcastParam, params of broadcast
 * \param [in] value, input value
 */
template<typename Traits, const auto& Policy, typename DataType = typename ATVC::TypeListGet<typename Traits::In::types, 0>::Type>
__global__ __aicore__ void AddcmulCustom(GM_ADDR tensor1,
                                         GM_ADDR tensor2,
                                         GM_ADDR input,
                                         GM_ADDR output,
                                         ATVC::BroadcastParam broadcastParam,
                                         DataType value)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 1. get input and output for kernel op from host Traits
    using KernelOpIn = typename Traits::In::types;
    using KernelOpOut = typename Traits::Out::types;
    using KernelOpTemp = typename Traits::Temp::types;

    // 2. define input and output for pre compute 
    using PreComputeInput = ATVC::OpInputs<typename ATVC::TypeListGet<KernelOpIn, 0>::Type, typename ATVC::TypeListGet<KernelOpIn, 1>::Type>;
    using PreComputeOutput = ATVC::OpOutputs<typename ATVC::TypeListGet<KernelOpTemp, 1>::Type>;
    using PreComputeTemp = ATVC::OpOutputs<typename ATVC::TypeListGet<KernelOpTemp, 0>::Type>;
    using PreComputeOpTraits =  ATVC::OpTraits<PreComputeInput, PreComputeOutput, PreComputeTemp>;
    using PreCompute = PreComputeMulOfBroadcast<PreComputeOpTraits>;

    // 3. define input and output for broadcast
    using BroadcastOpInput = ATVC::OpInputs<typename ATVC::TypeListGet<KernelOpTemp, 1>::Type>;
    using BroadcastOpOutput = ATVC::OpOutputs<typename ATVC::TypeListGet<KernelOpTemp, 2>::Type>;
    using BroadcastOpTraits =  ATVC::OpTraits<BroadcastOpInput, BroadcastOpOutput>;

    // 4. define input and output for post compute
    using PostComputeInput = ATVC::OpInputs<typename ATVC::TypeListGet<KernelOpTemp, 2>::Type, typename ATVC::TypeListGet<KernelOpIn, 2>::Type>;
    using PostComputeOutput = ATVC::OpOutputs<typename ATVC::TypeListGet<KernelOpOut, 0>::Type>;
    using PostComputeOpTraits = ATVC::OpTraits<PostComputeInput, PostComputeOutput>;
    using PostCompute = PostComputeAddOfBroadcast<PostComputeOpTraits>;

    // 5. call op run
    auto op = ATVC::Kernel::BroadcastOpTemplate<ATVC::BroadcastCompute<BroadcastOpTraits>, Policy, PreCompute, PostCompute>();
    ATVC::BroadcastParam *param = &broadcastParam;
    op.Run(tensor1, tensor2, input, output, param, value);
}
#endif
