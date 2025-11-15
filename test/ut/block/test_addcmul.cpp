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
#include "../../../examples/addcmul/addcmul.h"

using BroadcastOpTraits = ATVC::OpTraits<ATVC::OpInputs<float, float, float>, ATVC::OpOutputs<float>, ATVC::OpTemps<float, float, float>>;

void TestAddcmulCustom(uint32_t *tensor1, uint32_t *tensor2, uint32_t value, uint32_t *input, uint32_t *output, ATVC::BroadcastParam &param)
{
    AddcmulCustom<BroadcastOpTraits, ATVC::BROADCAST_POLICY1>(reinterpret_cast<uint8_t *>(tensor1), reinterpret_cast<uint8_t *>(tensor2), reinterpret_cast<uint8_t *>(input), reinterpret_cast<uint8_t *>(output), param, value);
}

class AtvcAddcmulTestsuite : public testing::Test {
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

TEST_F(AtvcAddcmulTestsuite, AtvcAddcmulTestCase)
{
    uint32_t eleNum = 8;
    uint32_t outEleNum = 8 * 8;
    uint32_t tensor1[eleNum] = {0};
    uint32_t tensor2[eleNum] = {0};
    uint32_t value = 3;
    uint32_t input[outEleNum] = {0};
    uint32_t output[outEleNum] = {0};
    uint32_t expectResult[outEleNum] = {};
    for (uint32_t i = 0; i < eleNum; i++) {
        tensor1[i] = i;
        tensor2[i] = i;
    }
    for (uint32_t i = 0; i < outEleNum; i++) {
        input[i] = i;
        expectResult[i] = i;
    }

    ATVC::BroadcastParam *param = new ATVC::BroadcastParam{};
    param->nBufferNum = 2;
    param->tilingData.A0 = 1;
    param->tilingData.A11 = 1;
    param->tilingData.A12 = 1;
    param->tilingData.A2 = 8;
    param->tilingData.B0 = 1;
    param->tilingData.B1 = 1;
    param->tilingData.B2 = 8;
    param->tilingData.coreNum = 1;
    param->tilingData.basicBlock = 32768;
    param->tilingData.factorACntPerCore = 8;
    param->tilingData.factorATotalCnt = 1;
    param->tilingData.factorBCntPerCore = 8;
    param->tilingData.factorBTotalCnt = 1;
    param->tilingData.shape[0] = 1;
    param->tilingData.shape[1] = 8;
    param->tilingData.dstShape[0] = 8;
    param->tilingData.dstShape[1] = 8;
    param->tilingData.stride[0] = 8;
    param->tilingData.stride[1] = 8;
    param->tilingData.dstStride[0] = 64;
    param->tilingData.dstStride[1] = 8;
    TestAddcmulCustom(tensor1, tensor2, value, input, output, *param);
    for (int i = 0; i < outEleNum; i++) {
        EXPECT_EQ(output[i], expectResult[i]);
    }
    delete param;
    AscendC::printf("AddcmulTestCase passed.\n");
}