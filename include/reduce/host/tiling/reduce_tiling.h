/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_REDUCE_TILING_H
#define ATVC_REDUCE_TILING_H
#include <algorithm>
#include <vector>
#include "graph/types.h"
#include "common/compile_info.h"
#include "common/const_def.h"
#include "common/ops_utils_host.h"
#include "tiling_common.h"

namespace OpTiling {
/*!
 * ReduceOpTiling: High-level tiling engine for Ascend Reduce kernels.
 * Computes cache-friendly UB blocks, multi-core split factors, loop counts and required workspace size.
 * Writes everything into the user-supplied `ReduceParam` and `ReducePolicy` structures so the runtime
 * can launch the kernel.
 */
template<class OpTraits, class PreComputeTraits, class PostComputeTraits>
class ReduceOpTiling {
public:
    using InputTypes = typename OpTraits::In::types;
    using OutputTypes = typename OpTraits::Out::types;
    using TempTypes = typename OpTraits::Temp::types;
    static constexpr size_t OP_INPUT_COUNT = ATVC::TypeListSize<InputTypes>::VALUE;
    static constexpr size_t OP_OUTPUT_COUNT = ATVC::TypeListSize<OutputTypes>::VALUE;
    static constexpr size_t OP_TEMP_COUNT = ATVC::TypeListSize<TempTypes>::VALUE;
    static constexpr size_t REDUCE_UB_NUM = OP_INPUT_COUNT + OP_OUTPUT_COUNT + OP_TEMP_COUNT;
    /*!
     * \brief constructor that binds the engine to user-supplied output structures and input descriptor.
     * \param[in] inputParam, compile-time description of the Reduce op
     * \param[out] policy, static policy of Reduce Template
     * \param[out] param, dynamic param of Reduce Template
     */
    ReduceOpTiling(ReduceTilingInputParam &inputParam, ATVC::ReducePolicy *policy, ATVC::ReduceParam *param,
        ATVC::Host::ReduceTilingHyperParam &hyperParam)
        : param_(param), policy_(policy), opInput_(inputParam), hyperParam_(hyperParam)
    {
        compileInfo_ = ATVC::GetOpCompileInfo();
        /* Built-in tiling only support allocate unified memory evenly, so all we need to know is the definition of
        the complete operator. If you wants to allocate UB memory unevenly, you need to know the definitions of
        reduce, pre-compute, post-compute separately, and extend tiling to support non-uniform distribution. */
        reduce_basic_num_ = REDUCE_UB_NUM * param->nBufferNum;
    };

    /*!
     * \brief Orchestrates the full tiling workflow; returns 0 on success, ‑1 otherwise.
     * \param void
     * \return 0 success, 1 failure
     */
    int32_t Run()
    {
        MakeWrapDim(opInput_.reduceShape, opInput_.reduceDim);
        if (IsAxesValid(opInput_.reduceShape, opInput_.reduceDim) == -1) {
            return -1;
        }
        std::vector<uint64_t> newShape(ATVC::MAX_DIM, 1);
        int32_t newShapeSize = 0;
        EliminateOne(opInput_.reduceShape, opInput_.reduceDim, newShape, newShapeSize);
        MergeAxis(opInput_.reduceDim, newShape, newShapeSize);
        if (!DoTiling(newShape, newShapeSize)) {
            printf("[ERROR]: [ATVC][Reduce][Tiling] Do tiling failed!\n");
            return -1;
        }
        CalcWorkSpace();
        return 0;
    }

private:
template <class Pattern>
void ComputeStride(std::vector<uint64_t>& shape)
{
    uint64_t accDim = 1;
    uint64_t ds = 1;
    for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
        param_->tilingData.stride[dim] = accDim;
        param_->tilingData.dstStride[dim] = ds;
        accDim *= shape[dim];
        if (IsAxisA<Pattern>(dim)) {
            ds *= shape[dim];
        }
    }
    double meanVar = static_cast<double>(1) / static_cast<double>(accDim / ds);
    param_->tilingData.outSize = ds;
    param_->tilingData.meanVar = static_cast<float>(meanVar);
}

template <class Pattern>
void SetTilingKey()
{
    uint64_t groupR = param_->tilingData.groupR;
    uint32_t aCount = 0;
    uint32_t rCount = 0;
    uint32_t innerACount = 0;
    uint32_t innerRCount = 0;
    if (groupR == 1) {
        // normal case
        aCount = (unitA_.idx - (Pattern::FirstA ? 0 : 1)) / ATVC::CONST2 + 1;
        innerRCount = (unitR_.idx - (Pattern::FirstA ? 1 : 0)) / ATVC::CONST2 + 1;
    } else if (groupR <= compileInfo_.vectorCoreNum / ATVC::CONST2) {
        // group case
        aCount = (unitA_.idx - (Pattern::FirstA ? 0 : 1)) / ATVC::CONST2 + 1;
        rCount = (unitR_.idx - (Pattern::FirstA ? 1 : 0)) / ATVC::CONST2 + 1;
        rCount = rCount + aCount;
    } else {
        // atomic case
        rCount = (unitR_.idx - (Pattern::FirstA ? 1 : 0)) / ATVC::CONST2 + 1;
        innerACount = (unitA_.idx - (Pattern::FirstA ? 0 : 1)) / ATVC::CONST2 + 1;
    }

    uint32_t innerID = Pattern::TailA ? 0 : 1;
    policy_->patternID = Pattern::ID * ATVC::CONST10 + innerID;
    policy_->loopARCount = aCount * ATVC::CONST10 + rCount;
    policy_->loopInnerARCount = innerACount * ATVC::CONST10 + innerRCount;
}

void CalcWorkSpace()
{
    size_t spaceSize = 0;
    uint64_t groupR = param_->tilingData.groupR;
    uint64_t outSize = param_->tilingData.outSize;
    int32_t size = ge::GetSizeByDataType(opInput_.promoteDtpye);
    if (groupR > 1) {
        spaceSize = OpsUtils::CeilAlign(outSize * compileInfo_.vectorCoreNum * size, compileInfo_.ubBlockSize);
    }
    param_->workspaceSize = ATVC::WORKSPACE_SIZE + spaceSize;
}

void EliminateOne(const std::vector<int64_t>& oriShape, std::vector<int64_t>& axes, std::vector<uint64_t>& shape,
                                  int32_t& shapeSize)
{
    int32_t dstIdx = 1;  // The first number in the shape is given as 1, so the first number is skipped.
    for (size_t i = 0; i < axes.size(); i++) {
        // The front dimension is filled, and all axes need to be increased by 1
        axes[i] = axes[i] + 1;
    }
    int32_t eraseNum = 0;
    for (size_t i = 0; i < oriShape.size(); i++) {
        auto iter = std::find(axes.begin(), axes.end(), i + 1);
        if (oriShape[i] != 1) {
            shape[dstIdx++] = oriShape[i];
            if (iter != axes.end()) {
                *iter = *iter - eraseNum;
            }
        } else {
            eraseNum++;
            if (iter != axes.end()) {
                axes.erase(iter);
            }
        }
    }
    shapeSize = dstIdx;
}

void MergeAxis(std::vector<int64_t>& axes, std::vector<uint64_t>& shape, int32_t& shapeSize)
{
    int32_t tmpSize = 0;
    for (int32_t i = 0; i < shapeSize;) {
        auto iter0 = std::find(axes.begin(), axes.end(), i);
        bool isRAxis0 = iter0 != axes.end();
        uint64_t s = shape[i];
        int32_t j = i + 1;
        for (; j < shapeSize; j++) {
            auto iter1 = std::find(axes.begin(), axes.end(), j);
            bool isRAxis1 = iter1 != axes.end();
            if (isRAxis0 != isRAxis1) {
                break;
            }
            s *= shape[j];
            if (isRAxis1) {
                // Continuous R-axis, need to erase the index of subsequent R-axis
                axes.erase(iter1);
            }
        }
        i = j;
        shape[tmpSize++] = s;
        if (isRAxis0) {
            *iter0 = tmpSize - 1;
        }
    }
    for (int32_t i = tmpSize; i < shapeSize; i++) {
        shape[i] = 0;
    }
    shapeSize = tmpSize;
}

bool DoTiling(std::vector<uint64_t>& shape, int32_t shapeSize)
{
    switch (shapeSize) {
        case ATVC::CONST1:
            return ComputeTiling<ATVC::ReducePattern::A>(shape);
            param_->tilingData.coreNum = 1; // Scenario A: No reduce, only one core copydata
        case ATVC::CONST2:
            return ComputeTiling<ATVC::ReducePattern::AR>(shape);
        case ATVC::CONST3:
            return ComputeTiling<ATVC::ReducePattern::ARA>(shape);
        case ATVC::CONST4:
            return ComputeTiling<ATVC::ReducePattern::ARAR>(shape);
        case ATVC::CONST5:
            return ComputeTiling<ATVC::ReducePattern::ARARA>(shape);
        case ATVC::CONST6:
            return ComputeTiling<ATVC::ReducePattern::ARARAR>(shape);
        case ATVC::CONST7:
            return ComputeTiling<ATVC::ReducePattern::ARARARA>(shape);
        case ATVC::CONST8:
            return ComputeTiling<ATVC::ReducePattern::ARARARAR>(shape);
        case ATVC::CONST9:
            return ComputeTiling<ATVC::ReducePattern::ARARARARA>(shape);
        default:
            return false;
    }
}

template <class Pattern>
bool ComputeExtraUnitA(const std::vector<uint64_t>& shape)
{
    // if RAxis is fully loaded in UB, and basicBlock is not enough, we need to calculate extra unitA
    if (unitR_.idx != -1 || Pattern::ID == ATVC::ReducePattern::PATTERN_A) {
        return true;
    }
    uint64_t innerR = unitR_.inner;
    uint64_t innerA = unitA_.inner / unitA_.step;  // restore original innerA
    uint64_t len = unitA_.idx == cBlock_.axis ? cBlock_.cacheLineOuter : shape[unitA_.idx];
    uint64_t outerA = unitA_.outer / OpsUtils::CeilDiv(len, unitA_.step) * len;
    uint64_t dTypeSize = ge::GetSizeByDataType(opInput_.inputDtype);
    uint64_t promoteDtypeSize = ge::GetSizeByDataType(opInput_.promoteDtpye);
    if (dTypeSize == 0 || promoteDtypeSize == 0) {
        printf("[ERROR]: [ATVC][Reduce][Tiling] Input dtype size cannot be zero!\n");
        return false;
    }
    uint64_t bBlockNum = basicBlock_ / dTypeSize;
    uint64_t maxInnerA = CACHE_SIZE / promoteDtypeSize;
    uint64_t step = 1;
    int32_t idxA;
    for (idxA = unitA_.idx; idxA > -1; idxA -= ATVC::CONST2) {
        uint64_t axislen = (idxA == cBlock_.axis ? cBlock_.cacheLineOuter : shape[idxA]);
        bool splitHere = false;
        for (step = (idxA == unitA_.idx ? unitA_.step : 1); step <= axislen; step += 1) {
            uint64_t tmpInnerA = innerA * step;
            uint64_t tmpOuterA = outerA / axislen * OpsUtils::CeilDiv(axislen, step);
            double rate = (double)tmpOuterA / (double)(OpsUtils::CeilAlign(tmpOuterA, compileInfo_.vectorCoreNum));
            bool isContinue = (rate > hyperParam_.balanceThreshHold && tmpInnerA * innerR * cBlock_.size <= bBlockNum &&
                               tmpInnerA * cBlock_.aSize < maxInnerA);
            if (isContinue) {
                continue;
            } else {
                step = step > 1 ? step - 1 : step;
                splitHere = true;
                break;
            }
        }
        if (splitHere) {
            innerA *= step;
            outerA = outerA / axislen * OpsUtils::CeilDiv(axislen, step);
            break;
        }
        innerA *= axislen;
        outerA /= axislen;
    }
    unitA_.Update(idxA, innerA, outerA, step);
    return true;
}

template <class Pattern>
void CalcBasicBlock()
{
    if (Pattern::ID == ATVC::ReducePattern::PATTERN_A) {
        basicBlock_ = hyperParam_.basicBlock;
    } else if (opInput_.inputDtype != opInput_.promoteDtpye) {
        basicBlock_ = hyperParam_.basicBlock / ATVC::CONST2;
    } else {
        basicBlock_ = hyperParam_.basicBlock;
    }
    if (basicBlock_ > OpsUtils::FloorAlign(compileInfo_.ubSize / reduce_basic_num_, ATVC::UB_ALIGN_32)) {
        basicBlock_ = OpsUtils::FloorAlign(compileInfo_.ubSize / reduce_basic_num_, ATVC::UB_ALIGN_32);
    }
}

template <class Pattern>
bool ComputeEmptyTiling(std::vector<uint64_t>& shape)
{
    uint64_t s = 1;
    for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
        if (IsAxisA<Pattern>(dim)) {
            s *= shape[dim];
        }
    }
    param_->tilingData.outSize = s;
    if (s == 0) {
        return true;
    }
    basicBlock_ = hyperParam_.basicBlock;
    if (basicBlock_ > OpsUtils::FloorAlign(compileInfo_.ubSize / reduce_basic_num_, ATVC::UB_ALIGN_32)) {
        basicBlock_ = OpsUtils::FloorAlign(compileInfo_.ubSize / reduce_basic_num_, ATVC::UB_ALIGN_32);
    }
    std::vector<uint64_t> newshape(ATVC::MAX_DIM, s);
    if (!CalcCacheLineStep<ATVC::ReducePattern::A>(newshape)) { return false; }
    unitA_.outer *= cBlock_.cacheLineOuter;
    ComputeUnitA<ATVC::ReducePattern::A>(newshape);
    ComputeSplit<ATVC::ReducePattern::A>(newshape);
    return true;
}

template <class Pattern>
bool ComputeTiling(std::vector<uint64_t>& shape)
{
    // Tiling can be divided into three steps
    // First: find the cacheLine-aligned axis from the innermost axis
    // Second: All the axes inside the CacheLine are a single unit
    //  1. The A-axis in UB is multiplied by the A-axis in CacheLine, and the result is less than one threshold(128)
    //  2. The A-axis outside UB, should ensure that can be cutted by enough cores
    //  3. The rest of the space for the basicblock is reserved for the R-axis
    CalcBasicBlock<Pattern>();
    if (IsEmtpyTensor<Pattern>(shape)) {
        return ComputeEmptyTiling<Pattern>(shape);
    }
    if (!CalcCacheLineStep<Pattern>(shape)) { return false; }
    int32_t axisInCacheLine = cBlock_.axis;
    for (int32_t i = Pattern::FirstA ? 0 : 1; i < axisInCacheLine; i += ATVC::CONST2) {
        unitA_.outer *= shape[i];
    }
    for (int32_t i = Pattern::FirstA ? 1 : 0; i < axisInCacheLine; i += ATVC::CONST2) {
        unitR_.outer *= shape[i];
    }

    bool basicSplitA = (axisInCacheLine + (Pattern::FirstA ? 1 : 0)) % ATVC::CONST2;
    if (basicSplitA) {
        unitA_.outer *= cBlock_.cacheLineOuter;
    } else {
        unitR_.outer *= cBlock_.cacheLineOuter;
    }
    ComputeUnitA<Pattern>(shape);
    ComputeUnitR<Pattern>(shape);

    if (!ComputeExtraUnitA<Pattern>(shape)) { return false; }

    ComputeSplit<Pattern>(shape);
    SetTilingKey<Pattern>();
    return true;
}

template <class Pattern>
void ComputeSplit(std::vector<uint64_t>& shape)
{
    uint64_t cacheStep = cBlock_.cacheLineStep;
    int32_t axis = cBlock_.axis;
    uint64_t perCoreNum = OpsUtils::CeilDiv(unitA_.outer * unitR_.outer, compileInfo_.vectorCoreNum);
    uint64_t blockDim = OpsUtils::CeilDiv(unitA_.outer * unitR_.outer, perCoreNum);
    uint64_t factorA = unitA_.idx == axis ? unitA_.step * cacheStep : unitA_.step;
    uint64_t factorR = unitR_.idx == axis ? unitR_.step * cacheStep : unitR_.step;

    uint64_t rStepNum = unitR_.idx < 0 ? 1 : OpsUtils::CeilDiv(shape[unitR_.idx], factorR);
    uint64_t totalOutNoStepR = unitR_.outer / rStepNum * unitA_.outer;
    for (auto iR = unitR_.idx; iR > -1;) {
        if (totalOutNoStepR < blockDim) {
            auto tmpBlockDim = OpsUtils::CeilAlign(blockDim, totalOutNoStepR);
            if (tmpBlockDim <= compileInfo_.vectorCoreNum) {
                blockDim = tmpBlockDim;
            } else {
                blockDim = OpsUtils::FloorAlign(blockDim, totalOutNoStepR);
            }
            break;
        }
        iR -= ATVC::CONST2;
        if (iR > 0) {
            totalOutNoStepR /= shape[iR];
        }
    }

    param_->tilingData.ubFactorA = factorA;
    uint64_t factorACntPerCore = OpsUtils::CeilDiv(unitA_.outer, blockDim);
    param_->tilingData.factorACntPerCore = factorACntPerCore;
    param_->tilingData.factorATotalCnt = unitA_.outer;

    param_->tilingData.ubFactorR = factorR;
    uint64_t factorRCntPerCore = OpsUtils::CeilDiv(unitR_.outer, OpsUtils::CeilDiv(blockDim, unitA_.outer));
    param_->tilingData.factorRCntPerCore = factorRCntPerCore;
    param_->tilingData.factorRTotalCnt = unitR_.outer;
    param_->tilingData.groupR = OpsUtils::CeilDiv(unitR_.outer, factorRCntPerCore);
    for (size_t i = 0; i < ATVC::MAX_DIM; i++) {
        param_->tilingData.shape[i] = shape[i];
    }
    param_->tilingData.basicBlock = basicBlock_;
    param_->tilingData.coreNum = static_cast<int32_t>(compileInfo_.vectorCoreNum);
    ComputeStride<Pattern>(shape);
    uint32_t realCore = OpsUtils::CeilDiv(unitA_.outer, factorACntPerCore) *
                        OpsUtils::CeilDiv(unitR_.outer, factorRCntPerCore);
    param_->tilingData.coreNum = realCore;
}

template <class Pattern>
bool CalcCacheLineStep(const std::vector<uint64_t>& shape)
{
    // find the cacheLine-aligned axis
    // cacheLineStep record cacheLine-aligned axis's shape, while left is cacheLineOuter
    uint64_t dTypeSize = ge::GetSizeByDataType(opInput_.inputDtype);
    if (dTypeSize == 0) {
        printf("[ERROR]: [ATVC][Reduce][Tiling] Input dtype size cannot be zero!\n");
        return false;
    }
    uint64_t cacheSize = compileInfo_.cacheLineSize / dTypeSize;
    uint64_t ubBlockSize = compileInfo_.ubBlockSize / dTypeSize;
    uint64_t cacheLineShape = 1;
    uint64_t cacheLineStep = 1;
    uint64_t cacheLineOuter = 1;
    uint64_t aInCacheLine = 1;
    for (int32_t i = Pattern::Dim - 1; i > -1; --i) {
        cacheLineShape *= shape[i];
        if (cacheLineShape > cacheSize) {
            cacheLineShape /= shape[i];
            cacheLineStep = OpsUtils::CeilDiv(cacheSize, cacheLineShape);
            cacheLineShape *= cacheLineStep;
            cacheLineOuter = OpsUtils::CeilDiv(shape[i], cacheLineStep);
            cBlock_.axis = i;
            break;
        } else {
            cacheLineStep = shape[i];
            cBlock_.axis = i;
        }
    }
    const int32_t startIndex = Pattern::TailA ? Pattern::Dim - 1 : Pattern::Dim - ATVC::CONST2;
    for (int32_t i = startIndex; i > cBlock_.axis; i -= ATVC::CONST2) {
        aInCacheLine *= (i == Pattern::Dim - 1) ? OpsUtils::CeilAlign(shape[i], ubBlockSize) : shape[i];
    }
    bool basicSplitA = (cBlock_.axis + (Pattern::FirstA ? 1 : 0)) % ATVC::CONST2;
    if (basicSplitA) {
        aInCacheLine *= cacheLineStep;
    }
    for (int32_t i = Pattern::Dim - 1; i > cBlock_.axis; --i) {
        cBlock_.size *= (i == Pattern::Dim - 1) ? OpsUtils::CeilAlign(shape[i], ubBlockSize) : shape[i];
    }
    cBlock_.size *= cacheLineStep;
    cBlock_.cacheLineStep = cacheLineStep;
    cBlock_.cacheLineOuter = cacheLineOuter;
    cBlock_.aSize = aInCacheLine;
    return true;
}

template <class Pattern>
void ComputeUnitA(const std::vector<uint64_t>& shape)
{
    int32_t axisInCacheLine = cBlock_.axis;
    uint64_t outerA = unitA_.outer;
    uint64_t innerA = unitA_.inner;
    uint64_t maxCacheA = hyperParam_.maxInnerA * sizeof(float) / ge::GetSizeByDataType(opInput_.promoteDtpye);
    const bool isPatternA = (Pattern::ID == ATVC::ReducePattern::PATTERN_A);
    const uint64_t patternABlockSize = basicBlock_ / ge::GetSizeByDataType(opInput_.inputDtype);
    uint64_t maxInnerA = isPatternA ? patternABlockSize : maxCacheA;
    uint64_t stepLen = Pattern::ID == ATVC::ReducePattern::PATTERN_A ? A_STEP_LEN : 1;
    bool basicSplitA = (axisInCacheLine + (Pattern::FirstA ? 1 : 0)) % ATVC::CONST2;
    uint64_t bBlockNum = basicBlock_ / ge::GetSizeByDataType(opInput_.inputDtype);
    uint64_t step = 1;
    int32_t iA;
    for (iA = basicSplitA ? axisInCacheLine : axisInCacheLine - 1; iA > -1; iA -= ATVC::CONST2) {
        uint64_t axislen = ((iA == axisInCacheLine) ? cBlock_.cacheLineOuter : shape[iA]);
        bool splitHere = false;
        uint64_t maxStep = 0;
        for (step = 1; step <= axislen / stepLen; step++) {
            uint64_t s = step * stepLen;
            uint64_t tmpInnerA = innerA * s;
            uint64_t tmpOuterA = outerA / axislen * OpsUtils::CeilDiv(axislen, s);
            double rate = (double)tmpOuterA / (double)(OpsUtils::CeilAlign(tmpOuterA, compileInfo_.vectorCoreNum));
            uint64_t aSize = tmpInnerA * cBlock_.aSize;
            if (iA == axisInCacheLine) {
                aSize = (cBlock_.aSize / cBlock_.cacheLineStep) * std::min(cBlock_.cacheLineStep * s, shape[iA]);
            }
            if (aSize <= maxInnerA && tmpInnerA * cBlock_.size <= bBlockNum) {
                maxStep = rate > hyperParam_.balanceThreshHold ? step : maxStep;
            } else {
                step = step > 1 ? step - 1 : step;
                splitHere = true;
                break;
            }
        }
        if (splitHere || maxStep <= 1 || iA - ATVC::CONST2 < 0) {
            step = maxStep == 0 ? 1 : maxStep * stepLen;
            innerA *= step;
            outerA = outerA / axislen * OpsUtils::CeilDiv(axislen, step);
            break;
        }
        innerA *= axislen;
        outerA /= axislen;
    }
    unitA_.Update(iA, innerA, outerA, step);
}

template <class Pattern>
void ComputeUnitR(std::vector<uint64_t>& shape)
{
    int32_t axisInCacheLine = cBlock_.axis;
    uint64_t outerR = unitR_.outer;
    uint64_t innerR = unitR_.inner;
    uint64_t outerA = unitA_.outer;
    uint64_t innerA = unitA_.inner;
    uint64_t step = 1;
    uint64_t bBlockNum = basicBlock_ / ge::GetSizeByDataType(opInput_.inputDtype);
    bool basicSplitA = (axisInCacheLine + (Pattern::FirstA ? 1 : 0)) % ATVC::CONST2;
    int32_t iR;
    for (iR = basicSplitA ? axisInCacheLine - 1 : axisInCacheLine; iR > -1; iR -= ATVC::CONST2) {
        uint64_t axislen = shape[iR];
        if (iR == axisInCacheLine) {
            axislen = cBlock_.cacheLineOuter;
        }
        innerR *= axislen;
        if (innerR * innerA * cBlock_.size <= bBlockNum) {
            outerR = outerR / axislen;
            continue;
        }

        innerR /= axislen;
        // maybe bBlockNum not full
        step = std::min(bBlockNum / (innerA * innerR * cBlock_.size), axislen);
        for (; step > 1; step--) {
            auto tmpOuterR = outerR / axislen * OpsUtils::CeilDiv(axislen, step);
            double rate = (double)(outerA * tmpOuterR) /
                          (double)(OpsUtils::CeilAlign(outerA * tmpOuterR, compileInfo_.vectorCoreNum));
            if (rate > hyperParam_.balanceThreshHold) {
                innerR *= step;
                outerR = outerR / axislen * OpsUtils::CeilDiv(axislen, step);
                break;
            }
        }
        break;
    }
    unitR_.Update(iR, innerR, outerR, step);
}

private:
    int32_t basicBlock_ = 0;
    ATVC::ReduceParam* param_ {nullptr};
    ATVC::ReducePolicy* policy_ {nullptr};
    ATVC::OpCompileInfo compileInfo_ = {0, 0, 0, 0};
    CacheLineBlock cBlock_;
    ReduceTilingUnit unitA_;
    ReduceTilingUnit unitR_;
    ReduceTilingInputParam opInput_;
    ATVC::Host::ReduceTilingHyperParam hyperParam_;
    uint32_t reduce_basic_num_;
};  // class ReduceOpTiling
}  // namespace OpTiling

#endif // ATVC_REDUCE_TILING_H