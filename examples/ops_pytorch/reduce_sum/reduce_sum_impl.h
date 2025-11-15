/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SUM_CUSTOM_IMPL_H
#define REDUCE_SUM_CUSTOM_IMPL_H
#include "reduce/host/reduce_host.h"
#include "reduce/kernel/reduce_kernel.h"

using ReduceOpTraitsFloat = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
using ReduceOpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int32_t>, ATVC::OpOutputs<int32_t>>;

/*
 * 该函数为ReduceSumCustom算子核函数入口
 * x                 Device上的gm地址，指向Add算子第一个输入
 * y                 Device上的gm地址，指向Add算子第一个输出
 * reduceParam       ATVC::ReduceParam
 */
template <typename Traits, const auto &Policy>
__global__ __aicore__ void ReduceSumCustom(GM_ADDR x, GM_ADDR y, ATVC::ReduceParam reduceParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);  // 使用了多核控制指令，设置算子执行时只启动Vector核
    // 将计算模板类模板定义作为模板参数传入，Policy由Host层的策略分派API给出
    auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<Traits>,
        Policy>();
    op.Run(x, y, &reduceParam);
}

#endif
