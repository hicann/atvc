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
#include "../../../examples/add_with_broadcast/add_with_broadcast.h"

using BroadcastOpTraits = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>, ATVC::OpTemps<float>>;
void TestBroadcastCustom(uint32_t *aGm, uint32_t *bGm, uint32_t *cGm, ATVC::BroadcastParam &bParam)
{
    AddWithBroadcastCustom<BroadcastOpTraits, ATVC::BROADCAST_POLICY1>(reinterpret_cast<uint8_t *>(aGm), reinterpret_cast<uint8_t *>(bGm), reinterpret_cast<uint8_t *>(cGm), bParam);
}

class AtvcAddWithBroadcastTestsuite : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }

    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

TEST_F(AtvcAddWithBroadcastTestsuite, AtvcAddWithBroadcastTestCase)
{
    uint32_t eleNum = 8;
    uint32_t outEleNum = 8 * 8;
    uint32_t aGm[eleNum] = {0};
    uint32_t bGm[outEleNum] = {0};
    uint32_t cGm[outEleNum] = {0};
    uint32_t expectResult[outEleNum] = {};
    for (uint32_t i = 0; i < eleNum; i++) {
        aGm[i] = i; // 初始化输入数据
    }
    for (uint32_t i = 0; i < outEleNum; i++) {
        expectResult[i] = i % 8; // 期望输出数据
    }

    ATVC::BroadcastParam *bParam = new ATVC::BroadcastParam{};
    bParam->nBufferNum = 2;
    bParam->tilingData.A0 = 1;
    bParam->tilingData.A11 = 1;
    bParam->tilingData.A12 = 1;
    bParam->tilingData.A2 = 8;
    bParam->tilingData.B0 = 1;
    bParam->tilingData.B1 = 1;
    bParam->tilingData.B2 = 8;
    bParam->tilingData.coreNum = 1;
    bParam->tilingData.basicBlock = 32768;
    bParam->tilingData.factorACntPerCore = 8;
    bParam->tilingData.factorATotalCnt = 1;
    bParam->tilingData.factorBCntPerCore = 8;
    bParam->tilingData.factorBTotalCnt = 1;
    bParam->tilingData.shape[0] = 1;
    bParam->tilingData.shape[1] = 8;
    bParam->tilingData.dstShape[0] = 8;
    bParam->tilingData.dstShape[1] = 8;
    bParam->tilingData.stride[0] = 8;
    bParam->tilingData.stride[1] = 8;
    bParam->tilingData.dstStride[0] = 64;
    bParam->tilingData.dstStride[1] = 8;
    TestBroadcastCustom(aGm, bGm, cGm, *bParam);
    for (int i = 0; i < outEleNum; i++) {
        EXPECT_EQ(cGm[i], expectResult[i]);
    }
    delete bParam;
    AscendC::printf("AddWithBroadcastTestCase passed.\n");
}