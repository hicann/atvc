/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "reduce_sum_impl.h"

namespace AscendC {
inline namespace reduce {
// 负责Reduce类算子的调度，选择对应的Policy最佳策略并执行Kernel函数
template <class OpTraits>
void ReduceOpAdapter(uint8_t *x, uint8_t *y, ATVC::ReduceParam param, ATVC::ReducePolicy &policy)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    // 将tiling api计算出的ReducePolicy转化为编译态参数并实例化相应的核函数
    if (policy == ATVC::REDUCE_POLICY0) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY0><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY1) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY1><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY2) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY2><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY3) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY3><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY4) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY4><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY5) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY5><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY6) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY6><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY7) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY7><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY8) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY8><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY9) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY9><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY10) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY10><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY11) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY11><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY12) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY12><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY13) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY13><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY14) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY14><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY15) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY15><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY16) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY16><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY17) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY17><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY18) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY18><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY19) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY19><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY20) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY20><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY21) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY21><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    } else if (policy == ATVC::REDUCE_POLICY22) {
        ReduceSumCustom<OpTraits, ATVC::REDUCE_POLICY22><<<param.tilingData.coreNum, nullptr, stream>>>(x, y, param);
    }
}
}  // namespace reduce
}  // namespace AscendC

namespace ascendc_reduce_ops {
at::Tensor op_reduce_sum(const at::Tensor &x, const std::vector<int64_t> &dim)
{
    std::vector<int64_t> shapeIn;
    std::vector<int64_t> shapeOut;
    ATVC::ReduceParam param;  // Reduce运行态参数，包含TilingData以及临时空间的相关信息
    ATVC::ReducePolicy policy = {-1, -1, -1};  // Reduce运行态参数，负责映射最适合的Reduce模板实现
    for (int32_t size : x.sizes()) {
        shapeIn.push_back(size);
        shapeOut.push_back(size);
    }
    for (const auto &i : dim) {
        shapeOut[i] = 1;
    }
    auto options = torch::TensorOptions().dtype(x.scalar_type()).device(x.device());
    at::Tensor y = at::empty(shapeOut, options);
    if (x.scalar_type() == at::kFloat) {
        // Host侧调用Tiling API完成相关运行态参数的运算
        (void)ATVC::Host::CalcReduceTiling<ReduceOpTraitsFloat>(shapeIn, dim, &policy, &param);
        // 调用Adapter调度接口，完成核函数的模板调用
        AscendC::ReduceOpAdapter<ReduceOpTraitsFloat>(
            (uint8_t *)(x.storage().data()), (uint8_t *)(y.storage().data()), param, policy);
    } else if (x.scalar_type() == at::kInt) {
        // Host侧调用Tiling API完成相关运行态参数的运算
        (void)ATVC::Host::CalcReduceTiling<ReduceOpTraitsInt>(shapeIn, dim, &policy, &param);
        // 调用Adapter调度接口，完成核函数的模板调用
        AscendC::ReduceOpAdapter<ReduceOpTraitsInt>(
            (uint8_t *)(x.storage().data()), (uint8_t *)(y.storage().data()), param, policy);
    }
    return y;
}

TORCH_LIBRARY(ascendc_ops, m)
{
    m.def("sum", &ascendc_reduce_ops::op_reduce_sum);
}
}  // namespace ascendc_reduce_ops