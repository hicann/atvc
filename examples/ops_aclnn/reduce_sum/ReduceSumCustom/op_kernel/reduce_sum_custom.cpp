/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce/kernel/reduce_kernel.h"

using ReduceOpTraits = ATVC::OpTraits<ATVC::OpInputs<DTYPE_X>, ATVC::OpOutputs<DTYPE_Y>>;

extern "C" __global__ __aicore__ void reduce_sum_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(ATVC::ReduceParam);
    GET_TILING_DATA(param, tiling);
    if (param.policyId == ATVC::REDUCE_POLICY0.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY0>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY1.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY1>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY2.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY2>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY3.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY3>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY4.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY4>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY5.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY5>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY6.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY6>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY7.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY7>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY8.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY8>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY9.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY9>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY10.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY10>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY11.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY11>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY12.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY12>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY13.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY13>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY14.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY14>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY15.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY15>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY16.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY16>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY17.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY17>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY18.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY18>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY19.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY19>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY20.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY20>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY21.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY21>();
        op.Run(x, y, &param);
    } else if (param.policyId == ATVC::REDUCE_POLICY22.ID) {
        auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<ReduceOpTraits>, ATVC::REDUCE_POLICY22>();
        op.Run(x, y, &param);
    }
}
