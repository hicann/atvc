/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "common/kernel_utils.h"
#include "reduce/kernel/reduce_op_template.h"
#include "reduce/common/reduce_common.h"
#include "reduce/kernel/reduce_sum_compute.h"

namespace ATVC {
using ReduceOpTraits = OpTraits<OpInputs<float>, OpOutputs<float>>;

template <typename Traits, const auto &Policy>
__global__ __aicore__ void ReduceSumCustom(GM_ADDR x, GM_ADDR y, ReduceParam* reduceParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    auto op = Kernel::ReduceOpTemplate<ReduceSumCompute<Traits>, Policy>();
    op.Run(x, y, reduceParam);
}

void TestReduceSumCustom(uint32_t *x, uint32_t *y, ReduceParam *param)
{
    ReduceSumCustom<ReduceOpTraits, REDUCE_POLICY2>(
        reinterpret_cast<uint8_t *>(x), reinterpret_cast<uint8_t *>(x), param);
}

class AtvcReduceSumTestsuite : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2U);
    }

    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

TEST_F(AtvcReduceSumTestsuite, AtvcReduceSumTestCase)
{
    uint32_t eleNum = 2U;
    uint32_t aGm[eleNum] = {0};
    uint32_t bGm[1] = {0};

    ReduceParam *param = new ReduceParam{};
    param->tilingData.factorACntPerCore = 1;
    param->tilingData.factorATotalCnt = 1;
    param->tilingData.ubFactorA = 1;
    param->tilingData.factorRCntPerCore = 1;
    param->tilingData.factorRTotalCnt = 1;
    param->tilingData.ubFactorR = 1;
    param->tilingData.groupR = 1;
    param->tilingData.outSize = 1;
    param->tilingData.basicBlock = 49152U;
    param->tilingData.coreNum = 1;
    param->tilingData.shape[0] = 1;
    param->tilingData.shape[1] = 2U;
    param->tilingData.stride[0] = 2U;
    param->tilingData.stride[1] = 1;
    TestReduceSumCustom(aGm, bGm, param);
    EXPECT_EQ(bGm[0], 0);
}
}  // namespace ATVC