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
#include "broadcast/kernel/broadcast_op_template.h"
#include "broadcast/kernel/broadcast_compute.h"
#include "broadcast/common/broadcast_common.h"

namespace ATVC {
using BroadcastOpTraits = OpTraits<OpInputs<float>, OpOutputs<float>>;

template <typename Traits, const auto &Policy>
__global__ __aicore__ void BroadcastCustom(GM_ADDR x, GM_ADDR y, BroadcastParam &broadcastParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto op = Kernel::BroadcastOpTemplate<BroadcastCompute<Traits>, Policy>();
    BroadcastParam *param = &broadcastParam;
    op.Run(x, y, param);
}

void TestBroadcastCustom(uint32_t *x, uint32_t *y, BroadcastParam &param)
{
    BroadcastCustom<BroadcastOpTraits, BROADCAST_POLICY1>(
        reinterpret_cast<uint8_t *>(x), reinterpret_cast<uint8_t *>(y), param);
}

class AtvcBroadcastToTestsuite : public testing::Test {
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

TEST_F(AtvcBroadcastToTestsuite, AtvcBroadcastToTestCase)
{
    uint32_t eleNum = 8U;
    uint32_t outEleNum = 64U;
    uint32_t aGm[eleNum] = {0};
    uint32_t bGm[outEleNum] = {0};

    BroadcastParam *param = new BroadcastParam{};
    param->nBufferNum = 2U;
    param->tilingData.A0 = 1;
    param->tilingData.A11 = 1;
    param->tilingData.A12 = 1;
    param->tilingData.A2 = 8U;
    param->tilingData.B0 = 1;
    param->tilingData.B1 = 1;
    param->tilingData.B2 = 8U;
    param->tilingData.coreNum = 1;
    param->tilingData.basicBlock = 32768U;
    param->tilingData.factorACntPerCore = 8U;
    param->tilingData.factorATotalCnt = 1;
    param->tilingData.factorBCntPerCore = 8U;
    param->tilingData.factorBTotalCnt = 1;
    param->tilingData.shape[0] = 1;
    param->tilingData.shape[1] = 8U;
    param->tilingData.dstShape[0] = 8U;
    param->tilingData.dstShape[1] = 8U;
    param->tilingData.stride[0] = 8U;
    param->tilingData.stride[1] = 8U;
    param->tilingData.dstStride[0] = 64U;
    param->tilingData.dstStride[1] = 8U;
    TestBroadcastCustom(aGm, bGm, *param);
    for (int i = 0; i < outEleNum; i++) {
        EXPECT_EQ(bGm[0], 0);
    }
    delete param;
}
}  // namespace ATVC