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
 * \file elewise_host.h
 * \brief
 */

#ifndef ATVC_ELE_WISE_HOST_H
#define ATVC_ELE_WISE_HOST_H
#include <vector>
#include <cstdint>
#include "common/compile_info.h"
#include "common/atvc_opdef.h"
#include "common/atvc_op_check.h"
#include "elewise/common/elewise_common.h"

namespace ATVC {
namespace Host {
constexpr uint32_t BASIC_CNT_MIN = 32;  // Min number of basicBlock elements, because 32B alignment is required in UB
constexpr uint32_t MAX_SHAPE_NODE = 3;  // number of Data segmentation node
/*!
 * EleWiseTilingHyperParam: HyperParam for elementwise tiling optimization.
 */
struct EleWiseTilingHyperParam {
    uint32_t singleCoreBaseLine = 512;   // data volume baseline for a core, the valid setting range: [256, 128 * 1024]
    float ubSizeLimitThreshold = 0.95f;  // UB memory usage upper limit，determines the maximum value of basicBlock
    uint32_t nBufferNum = 2;             // The number of parallelism buffer, the valid setting range: [1, 2]
    uint32_t splitDataShape[MAX_SHAPE_NODE] = {1024, 32 * 1024, 64 * 1024};  // Segmentation nodes for shape
    uint32_t dataSplitFactor[MAX_SHAPE_NODE + 1] = {4, 4, 8, 6};  // The split coefficient for each segmentation nodes
    uint32_t rsvLiveCnt = 0;  // Additional surviving nodes, means to reserve a portion of UB space.
};
/*!
 * \brief Validate the legitimacy of elementwise tiling hyper param
 * \param[in] hyperParam, elementwise tiling hyper param
 * \return bool result, return true if the hyper param is valid, otherwise return false
 */
bool CheckEleWiseHyperParam(const EleWiseTilingHyperParam &hyperParam)
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
        printf("[ERROR]: [ATVC][EleWise] Tiling hyperParam is invalid: "
            "singleCoreBaseLine(%u) must be in [256, 128 * 1024].\n", hyperParam.singleCoreBaseLine);
        return false;
    }
    if(hyperParam.ubSizeLimitThreshold > MAX_UB_LIMIT || hyperParam.ubSizeLimitThreshold < MIN_UB_LIMIT) {
        printf("[ERROR]: [ATVC][EleWise] Tiling hyperParam is invalid: "
            "ubSizeLimitThreshold(%f) must be in [0.5, 0.96].\n", hyperParam.ubSizeLimitThreshold);
        return false;
    }
    if(hyperParam.nBufferNum > MAX_BUF_NUM || hyperParam.nBufferNum < MIN_BUF_NUM) {
        printf("[ERROR]: [ATVC][EleWise] Tiling hyperParam is invalid: nBufferNum(%u) must be in [1, 2].\n",
            hyperParam.nBufferNum);
        return false;
    }
    if(hyperParam.rsvLiveCnt > MAX_LIVE_CNT) {
        printf("[ERROR]: [ATVC][EleWise] Tiling hyperParam is invalid: rsvLiveCnt(%u) must be smaller than 2.\n",
            hyperParam.rsvLiveCnt);
        return false;
    }
    for (uint32_t i = 0; i < MAX_SHAPE_NODE - 1; i++) {
        if (hyperParam.splitDataShape[i] >= hyperParam.splitDataShape[i + 1]) {
            printf("[ERROR]: [ATVC][EleWise] Tiling hyperParam is invalid: "
                "splitDataShape[%u] = %u must be smaller than splitDataShape[%u] = %u.\n",
                i, hyperParam.splitDataShape[i], i + 1, hyperParam.splitDataShape[i + 1]);
            return false;
        }
    }

    for (uint32_t i = 0; i <= MAX_SHAPE_NODE; i++) {
        if (hyperParam.dataSplitFactor[i] == 0 || hyperParam.dataSplitFactor[i] > MAX_FACTOR_VAL) {
            printf("[ERROR]: [ATVC][EleWise] Tiling hyperParam is invalid: "
                "dataSplitFactor[%u]  must be in [1, 32]. but real value is %u\n", i, hyperParam.dataSplitFactor[i]);
            return false;
        }
    }
    return true;
}
/*!
 * \brief Calculate the base element count per block
 * \param[in] hyperParam, elementwise tiling hyper param
 * \param[in] totalCnt, the total number of elements in a single input
 * \param[in] blockNum, numbers of blocks
 * \param[in] ubufLimitCnt, ub upper constraints
 * \return int32_t the base element count per block
 */
int32_t GetEleWiseBasicCnt(
    const EleWiseTilingHyperParam &hyperParam, int32_t totalCnt, uint32_t blockNum, uint32_t ubufLimitCnt)
{
    uint32_t basicCnt = 0;
    if (blockNum == 0) {
        return 0;
    }
    uint32_t avgElePerBlock = totalCnt / blockNum;
    for (uint32_t i = 0; i < MAX_SHAPE_NODE; i++) {
        if (avgElePerBlock <= hyperParam.splitDataShape[i]) {
            basicCnt = avgElePerBlock / hyperParam.dataSplitFactor[i];
            break;
        }
    }
    if (avgElePerBlock > hyperParam.splitDataShape[MAX_SHAPE_NODE - 1]) {
        basicCnt = avgElePerBlock / hyperParam.dataSplitFactor[MAX_SHAPE_NODE];
    }
    // basicCnt must be smaller than the upper limit.
    if (basicCnt > ubufLimitCnt) {
        basicCnt = ubufLimitCnt;
    }
    // Ensure alignment of UB data block 32B
    basicCnt = basicCnt / BASIC_CNT_MIN * BASIC_CNT_MIN;
    // Control basicCnt to not be less than the minimum data size
    if (basicCnt < BASIC_CNT_MIN) {
        basicCnt = BASIC_CNT_MIN;
    }
    return basicCnt;
}

/*!
 * \brief Calculate the operational parameters of EleWiseParam for EleWise
 * \param[in] totalCnt, the total number of elements in a single input
 * \param[out] param, output parameters.
 * \return Return true to indicate calculation success, false to indicate failure.
 */
template <class OpTraits>
bool CalcEleWiseTiling(
    int32_t totalCnt, ATVC::EleWiseParam &param, EleWiseTilingHyperParam hyperParam = EleWiseTilingHyperParam())
{
    if (!CheckEleWiseHyperParam(hyperParam)) {
        return false;
    }
    using Inputs = typename OpTraits::In::types;
    using Outputs = typename OpTraits::Out::types;
    using Temps = typename OpTraits::Temp::types;
    // xxTensroSumbytes represents the cumulative length of all data types in tensorList,
    static constexpr size_t IN_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Inputs, SizeValue<0>, SumSizes>::Type::VALUE;
    static constexpr size_t OUT_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Outputs, SizeValue<0>, SumSizes>::Type::VALUE;
    static constexpr size_t TEMP_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Temps, SizeValue<0>, SumSizes>::Type::VALUE;
    uint32_t tensorSumBytes =
        (IN_TENSOR_SUM_BYTES + OUT_TENSOR_SUM_BYTES) * hyperParam.nBufferNum + TEMP_TENSOR_SUM_BYTES;
    if (hyperParam.rsvLiveCnt > 0) {
        tensorSumBytes =
            (IN_TENSOR_SUM_BYTES + OUT_TENSOR_SUM_BYTES) * (hyperParam.nBufferNum + 1) + TEMP_TENSOR_SUM_BYTES;
    }
    auto compileInfo = ATVC::GetOpCompileInfo();
    uint32_t ubSize = compileInfo.ubSize;
    uint32_t blockNum = (totalCnt < hyperParam.singleCoreBaseLine) ? 1 : totalCnt / hyperParam.singleCoreBaseLine;
    if (blockNum > compileInfo.vectorCoreNum) {
        blockNum = compileInfo.vectorCoreNum;
    }

    uint32_t ubufLimitCnt = (uint32_t)(ubSize * hyperParam.ubSizeLimitThreshold) / tensorSumBytes;

    int32_t basicCnt = GetEleWiseBasicCnt(hyperParam, totalCnt, blockNum, ubufLimitCnt);
    if (basicCnt == 0 || blockNum == 0) {
        printf("[ERROR]: [ATVC][EleWise] Tiling Error: initial basic count and block number can not be zero!\n");
        return false;
    }

    param.tilingData.tiledCnt = basicCnt;
    param.totalCnt = totalCnt;
    uint32_t totalCopyCnt = totalCnt / basicCnt;
    param.tilingData.tailBlockCnt = (totalCopyCnt) % blockNum;
    param.tilingData.blockNum = blockNum;
    param.tilingData.numPerBlock = totalCopyCnt / blockNum;  // The number of basic blocks to be transported per block
    param.tilingData.tailElemCnt = totalCnt % basicCnt;      // The number of tail block elements
    param.nBufferNum = hyperParam.nBufferNum;
    return true;
};
}  // namespace Host
}  // namespace ATVC
#endif  // ATVC_ELE_WISE_HOST_H