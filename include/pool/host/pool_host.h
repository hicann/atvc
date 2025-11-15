/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_host.h
 * \brief
 */

#ifndef ATVC_POOL_HOST_H
#define ATVC_POOL_HOST_H
#include <vector>
#include <cstdint>
#include <cmath>
#include "common/compile_info.h"
#include "common/atvc_opdef.h"
#include "common/atvc_op_check.h"
#include "pool/common/pool_common.h"

namespace ATVC {
namespace Host {
constexpr uint32_t BASIC_CNT_MIN = 32; // Min number of basicBlock elements, because 32B alignment is required in UB
constexpr uint32_t MAX_SHAPE_NODE = 3; // number of Data segmentation node

struct PoolTilingHyperParam {
    uint32_t singleCoreBaseLine = 512; // data volume baseline for a core, the valid setting range: [256, 128 * 1024]
    float ubSizeLimitThreshold = 0.95f; // UB memory usage upper limit，determines the maximum value of basicBlock
    uint32_t nBufferNum = 2;            // The number of parallelism buffer, the valid setting range: [1, 2]
    uint32_t splitDataShape[MAX_SHAPE_NODE] = {1024, 32 * 1024, 64 * 1024}; // Segmentation nodes for shape
    uint32_t dataSplitFactor[MAX_SHAPE_NODE + 1] = {4, 4, 8, 6}; // The split coefficient for each segmentation nodes
    uint32_t rsvLiveCnt = 0; // Additional surviving nodes, means to reserve a portion of UB space.
};

bool CheckPoolHyperParam(const PoolTilingHyperParam &hyperParam)
{
    constexpr uint32_t MAX_BASE_LINE = 128 *1024U;
    constexpr uint32_t MIN_BASE_LINE = 256U;
    constexpr float MAX_UB_LIMIT = 0.96f;
    constexpr float MIN_UB_LIMIT = 0.5f;
    constexpr uint32_t MAX_BUF_NUM = 2U;
    constexpr uint32_t MIN_BUF_NUM = 1U;
    constexpr uint32_t MAX_LIVE_CNT = 1U;
    constexpr uint32_t MAX_FACTOR_VAL = 32U;
    if(hyperParam.singleCoreBaseLine > MAX_BASE_LINE || hyperParam.singleCoreBaseLine < MIN_BASE_LINE) {
        printf("[ERROR]: [ATVC][Pool] Tiling hyperParam is invalid: "
            "singleCoreBaseLine(%u) must be in [256, 128 * 1024].\n", hyperParam.singleCoreBaseLine);
        return false;
    }
    if(hyperParam.ubSizeLimitThreshold > MAX_UB_LIMIT || hyperParam.ubSizeLimitThreshold < MIN_UB_LIMIT) {
        printf("[ERROR]: [ATVC][Pool] Tiling hyperParam is invalid: "
            "ubSizeLimitThreshold(%f) must be in [0.5, 0.96].\n", hyperParam.ubSizeLimitThreshold);
        return false;
    }
    if(hyperParam.nBufferNum > MAX_BUF_NUM || hyperParam.singleCoreBaseLine < MIN_BUF_NUM) {
        printf("[ERROR]: [ATVC][Pool] Tiling hyperParam is invalid: nBufferNum(%u) must be in [1, 2].\n",
            hyperParam.nBufferNum);
        return false;
    }
    if(hyperParam.rsvLiveCnt > MAX_LIVE_CNT) {
        printf("[ERROR]: [ATVC][Pool] Tiling hyperParam is invalid: rsvLiveCnt(%u) must be smaller than 2.\n",
            hyperParam.rsvLiveCnt);
        return false;
    }
    for (uint32_t i = 0; i < MAX_SHAPE_NODE - 1; i++) {
        if (hyperParam.splitDataShape[i] >= hyperParam.splitDataShape[i + 1]) {
            printf("[ERROR]: [ATVC][Pool] Tiling hyperParam is invalid: "
                "splitDataShape[%u] = %u must be smaller than splitDataShape[%u] = %u.\n",
                i, hyperParam.splitDataShape[i], i + 1, hyperParam.splitDataShape[i + 1]);
            return false;
        }
    }

    for (uint32_t i = 0; i <= MAX_SHAPE_NODE; i++) {
        if (hyperParam.dataSplitFactor[i] == 0 || hyperParam.dataSplitFactor[i] > MAX_FACTOR_VAL) {
            printf("[ERROR]: [ATVC][Pool] Tiling hyperParam is invalid: "
                "dataSplitFactor[%u]  must be in [1, 32]. but real value is %u\n", i, hyperParam.dataSplitFactor[i]);
            return false;
        }
    }
    return true;
}

/*!
 * \brief Calculate the operational parameters of EleWiseParam for EleWise
 * \param[in] totalLayout, the total height&width elements in a single input
 * \param[in] tileLayout, the tile height&width elements in a single input
 * \param[out] param, output parameters.
 * \return Return true to indicate calculation success, false to indicate failure.
 */
template <class OpTraits>
bool CalcPoolTiling(const ATVC::Layout2Dim &totalLayout, const ATVC::Layout2Dim &tileLayout, ATVC::PoolParam &param)
{
    using Inputs = typename OpTraits::In::types;
    using Outputs = typename OpTraits::Out::types;
    using Temps = typename OpTraits::Temp::types;

    static constexpr size_t IN_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Inputs, SizeValue<0>, SumSizes>::Type::VALUE;
    static constexpr size_t OUT_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Outputs, SizeValue<0>, SumSizes>::Type::VALUE;
    static constexpr size_t TEMP_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Temps, SizeValue<0>, SumSizes>::Type::VALUE;
    uint32_t totalH = totalLayout.height;
    uint32_t totalW = totalLayout.width;
    uint32_t tileH = tileLayout.height;
    uint32_t tileW = tileLayout.width;
    uint32_t totalCnt = totalH * totalW;
    auto compileInfo = ATVC::GetOpCompileInfo();
    uint32_t ubSize = compileInfo.ubSize;
    uint32_t tileNum = ((totalH + tileH - 1) / tileH) * ((totalW + tileW -1) / tileW);
    param.tilingData.blockNum = tileNum;
    PoolTilingHyperParam hyperParam{};
    if (!CheckPoolHyperParam(hyperParam)) {
        return false;
    }
    if (tileNum > compileInfo.vectorCoreNum) {
        param.tilingData.blockNum = compileInfo.vectorCoreNum;
    }

    int32_t basicCnt = tileH * tileW;
    if (basicCnt == 0 || param.tilingData.blockNum == 0) {
        printf("[ERROR]: [ATVC][Pool] Tiling Error: initial basic count and block number can not be zero!\n");
        return false;
    }

    param.tilingData.tiledCnt = basicCnt;
    param.totalCnt = totalCnt;
    param.tilingData.numPerBlock = tileNum / param.tilingData.blockNum;
    param.tilingData.tailBlockCnt = tileNum % param.tilingData.blockNum;
    param.nBufferNum = hyperParam.nBufferNum;
    //  ------------------------
    param.tilingData.height = totalH;
    param.tilingData.width = totalW;
    param.tilingData.tiledCntH = tileH;
    param.tilingData.tiledCntW = tileW;
    return true;
};
}
} // namespace ATVC
#endif // ATVC_POOL_HOST_H