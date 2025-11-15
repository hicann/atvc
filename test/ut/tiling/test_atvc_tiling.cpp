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
#include "elewise/common/elewise_common.h"
#include "elewise/host/elewise_host.h"
#include "reduce/common/reduce_common.h"
#include "reduce/host/reduce_host.h"
#include "broadcast/common/broadcast_common.h"
#include "broadcast/host/broadcast_host.h"
#include "platform_stub.h"

class TestAtvcTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
    virtual void SetUp()
    {}
    void TearDown()
    {}
};

TEST_F(TestAtvcTiling, TestAtvcTilingEleWiseAddCase)
{
    using ADD_OPTRAITS = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;
    int32_t eleNum = 8 * 1024;
    ATVC::EleWiseParam param;

    ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param);

    EXPECT_EQ(param.nBufferNum, 2);
}

TEST_F(TestAtvcTiling, TestAtvcTilingEleWiseAddHyperParamError)
{
    using ADD_OPTRAITS = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;
    int32_t eleNum = 8 * 1024; 
    ATVC::EleWiseParam param;
    ATVC::Host::EleWiseTilingHyperParam hyperParam;
    hyperParam.singleCoreBaseLine = 255; // set base count for single core(255).
    auto ret = ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.singleCoreBaseLine = 128 *1024 + 1; // set base count for single core(128 *1024 + 1).
    ret = ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.singleCoreBaseLine = 1024;
    hyperParam.ubSizeLimitThreshold = 0.97f;
    ret = ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.ubSizeLimitThreshold = 0.49f;
    ret = ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.ubSizeLimitThreshold = 0.95f;
    hyperParam.splitDataShape[0] = 33 * 1024;
    ret = ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.splitDataShape[0] = 1024;
    hyperParam.dataSplitFactor[0] = 0;
    ret = ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.dataSplitFactor[0] = 2;
    hyperParam.nBufferNum = 0; // set nBufferNum 0
    ret = ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.nBufferNum = 3; // set nBufferNum 3
    ret = ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
     hyperParam.nBufferNum = 2; // set nBufferNum 2
    ret = ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param, hyperParam=hyperParam);
    EXPECT_EQ(ret, true);
}

TEST_F(TestAtvcTiling, TestAtvcTilingReduceSumCase)
{
    using ReduceOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
    std::vector<int64_t> dim{0};
    std::vector<int64_t> shape{8, 1024};
    ATVC::ReduceParam param;
    ATVC::ReducePolicy policy = {-1, -1, -1};
    auto ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param);
    EXPECT_EQ(ret, true);

    dim = {1, 2};
    shape = {256, 682, 553};
    ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param);
    EXPECT_EQ(ret, true);

    dim = {3};
    shape = {128, 38, 38, 1};
    ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param);
    EXPECT_EQ(ret, true);

    dim = {0, 1, 2};
    shape = {256, 1178, 300};
    ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param);
    EXPECT_EQ(ret, true);

    dim = {0, 1, 2,3};
    shape = {4, 176, 64, 2000};
    ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param);
    EXPECT_EQ(ret, true);

    dim = {0};
    shape = {7, 6831, 11, 2, 11, 3, 2};
    ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param);
    EXPECT_EQ(ret, true);
    dim = {1};
    shape = {7, 6831, 11, 2, 11, 3, 2};
    ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param);
    EXPECT_EQ(ret, true);
}

TEST_F(TestAtvcTiling, TestAtvcTilingReduceSumHyperParamError)
{
    using ReduceOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
    std::vector<int64_t> dim{0};
    std::vector<int64_t> shape{8, 1024};
    ATVC::ReduceParam param;
    ATVC::ReducePolicy policy = {-1, -1, -1};
    ATVC::Host::ReduceTilingHyperParam hyperParam;
    hyperParam.basicBlock = 54 * 1024 + 1;
    auto ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.basicBlock = 48 * 1024;
    hyperParam.maxInnerA = 257;
    ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.maxInnerA = 128;
    hyperParam.balanceThreshHold = 0.79f;
    ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param, hyperParam=hyperParam);
    EXPECT_EQ(ret, false);
    hyperParam.balanceThreshHold = 0.95f;
    ret = ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param, hyperParam=hyperParam);
    EXPECT_EQ(ret, true);
}

TEST_F(TestAtvcTiling, TestAtvcTilingBroadcastToCase)
{
    using BroadcastOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
    std::vector<int64_t> shapeIn{1, 1024};
    std::vector<int64_t> shapeOut{8, 1024};
    ATVC::BroadcastParam param;
    ATVC::BroadcastPolicy policy = {-1, -1, -1};
    auto ret = ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param);
    EXPECT_EQ(ret, true);

    shapeIn = {8, 1};
    shapeOut = {8, 8};
    ret = ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param);
    EXPECT_EQ(ret, true);

    shapeIn = {8, 1};
    shapeOut = {8, 2048};
    ret = ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param);
    EXPECT_EQ(ret, true);

    shapeIn = {2048, 1};
    shapeOut = {2048, 2048};
    ret = ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param);
    EXPECT_EQ(ret, true);

    shapeIn = {1, 2048};
    shapeOut = {1784, 2048};
    ret = ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param);
    EXPECT_EQ(ret, true);

    shapeIn = {2048, 1, 1};
    shapeOut = {2048, 12, 6};
    ret = ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param);
    EXPECT_EQ(ret, true);

    shapeIn = {1, 23, 83, 2, 1024};
    shapeOut = {8, 23, 83, 2, 1024};
    ret = ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param);
    EXPECT_EQ(ret, true);
}
