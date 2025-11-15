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
#include "add_custom_impl.h"

namespace ascendc_elewise_ops {
at::Tensor op_add_custom(const at::Tensor &x, const at::Tensor &y)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty_like(x);
    int32_t totalLength = 1;
    for (int32_t size : x.sizes()) {
        totalLength *= size;
    }
    // 声明运行态参数param
    ATVC::EleWiseParam param;
    if (x.scalar_type() == at::kFloat) {
        // Host侧调用Tiling API完成相关运行态参数的运算
        (void)ATVC::Host::CalcEleWiseTiling<AddOpTraitsFloat>(totalLength, param);
        // 调用核函数
        AddCustom<AddOpTraitsFloat><<<param.tilingData.blockNum, nullptr, stream>>>(
            (uint8_t *)(x.storage().data()), (uint8_t *)(y.storage().data()), (uint8_t *)(z.storage().data()), param);
    } else if (x.scalar_type() == at::kInt) {
        (void)ATVC::Host::CalcEleWiseTiling<AddOpTraitsInt>(totalLength, param);
        // 调用核函数
        AddCustom<AddOpTraitsInt><<<param.tilingData.blockNum, nullptr, stream>>>(
            (uint8_t *)(x.storage().data()), (uint8_t *)(y.storage().data()), (uint8_t *)(z.storage().data()), param);
    }
    return z;
}

TORCH_LIBRARY(ascendc_ops, m)
{
    m.def("add", &ascendc_elewise_ops::op_add_custom);
}
}  // namespace ascendc_elewise_ops