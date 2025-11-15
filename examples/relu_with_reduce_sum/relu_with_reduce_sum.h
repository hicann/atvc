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
#include "pre_compute_add_with_reduce_sum.h"
#include "post_compute_relu_with_reduce_sum.h"

/* !
 * \brief ReluWithReduceSum(i) = relu(reduce_sum(x+1))
 * \param [in] x, input global memory of x
 * \param [out] y, output global memory
 * \param [in] reduceParam, params of reduce_sum
 */
template <typename Traits, const auto &Policy>
__global__ __aicore__ void ReluWithReduceSum(GM_ADDR x,
                                             GM_ADDR y,
                                             ATVC::ReduceParam reduceParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    // 1. get input and output for kernel op from host Traits
    using KernelOpInput = typename Traits::In::types;
    using KernelOpOutput = typename Traits::Out::types;
    using KernelOpTemp = typename Traits::Temp::types;

    // 2. define input and output for pre compute
    using PreComputeInput = ATVC::OpInputs<typename ATVC::TypeListGet<KernelOpInput, 0>::Type>;
    using PreComputeOutput = ATVC::OpOutputs<typename ATVC::TypeListGet<KernelOpTemp, 0>::Type>;
    using PreComputeOpTraits = ATVC::OpTraits<PreComputeInput, PreComputeOutput>;
    using PreCompute = PreComputeAddOfReduceSum<PreComputeOpTraits>;

    // 3. define input and output for reduce_sum
    using ReduceSumOpInput = ATVC::OpInputs<typename ATVC::TypeListGet<KernelOpTemp, 0>::Type>;
    using ReduceSumOpOutput = ATVC::OpOutputs<typename ATVC::TypeListGet<KernelOpTemp, 0>::Type>;
    using ReduceSumOpTraits = ATVC::OpTraits<ReduceSumOpInput, ReduceSumOpOutput>;

    // 4. define input and output for post compute
    using PostComputeInput = ATVC::OpInputs<typename ATVC::TypeListGet<KernelOpTemp, 0>::Type>;
    using PostComputeOutput = ATVC::OpOutputs<typename ATVC::TypeListGet<KernelOpOutput, 0>::Type>;
    using PostComputeOpTraits = ATVC::OpTraits<PostComputeInput, PostComputeOutput>;
    using PostCompute = PostComputeReluOfReduceSum<PostComputeOpTraits>;

    // 5. call op run
    auto op = ATVC::Kernel::ReduceOpTemplate<
        ATVC::ReduceSumCompute<ReduceSumOpTraits>, 
        Policy, 
        PreCompute,
        PostCompute>();
    op.Run(x, y, &reduceParam);
}
#endif
