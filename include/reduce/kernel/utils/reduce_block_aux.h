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
 * \file reduce_block_aux.h
 * \brief reduce schedule aux
 */

#ifndef ATVC_REDUCE_BLOCK_AUX_H
#define ATVC_REDUCE_BLOCK_AUX_H
#include "common/kernel_utils.h"
#include "common/const_def.h"
#include "reduce/common/patterns.h"
#include "reduce_block_aux_util.h"

namespace {
template <int32_t dim>
struct SliceView {
    uint64_t addr;
    uint64_t burstLen;
    uint64_t axisSize;
    struct {
        uint64_t repeat = 1;
        uint64_t srcStride = 0;
        uint64_t dstStride = 0;
        uint64_t idx = 0;
        bool isAxisA = false;
    } axis[dim];
};
}
namespace ATVC {
namespace KernelUtils {
namespace Reduce {
template <typename DataType, auto LoopInfo, class ReduceOp, class InDtype, class OutDtype, class PreOp = void,
    class EOp = void>
struct ReduceBlockAux {
    using Pattern = typename ReducePattern::GetPattern<LoopInfo->patternID>::T;
    using InnerPattern = typename ReducePattern::GetPattern<LoopInfo->innerPatternID>::T;
    using PromoteDataType = typename ReduceOp::PromoteDataType;
    constexpr static int32_t DIM = Pattern::Dim;
    constexpr static uint32_t INPUT_SIZE = ReduceOp::InputSize;
    constexpr static uint32_t OUTPUT_SIZE = ReduceOp::OutputSize;
    constexpr static int32_t ELEMENT_ONE_REPEAT = Platform::GetVRegSize() / sizeof(PromoteDataType);
    constexpr static uint64_t BLOCK_SIZE_BYTE = Platform::GetUbBlockSize();
    static constexpr bool BLOCK_PRE_COMPUTE = !AscendC::Std::is_same_v<PreOp, void>;
    static constexpr bool BLOCK_POST_COMPUTE = !AscendC::Std::is_same_v<EOp, void>;

public:
    uint64_t loopAStartIndex;
    uint64_t loopAEndIndex;
    uint64_t loopAAxisStep;
    uint64_t ubFactorA;

    uint64_t loopRStartIndex;
    uint64_t loopREndIndex;
    uint64_t loopRAxisStep;
    uint64_t ubFactorR;

    AscendC::GlobalTensor<InDtype>* input;
    AscendC::GlobalTensor<OutDtype>* output;

    int64_t rCount;
    int64_t bisectionPos;
    int64_t cacheCount;
    int64_t bisectionTail;

    uint64_t aOutBurstLen;
    uint64_t aOutNBurst;

    struct {
        uint64_t start = 0;
        uint64_t stride = 1; // Copy step size
    } iterAddr_[DIM];

    const DataType* tiling_;
    ReduceOp* op_;

public:
    __aicore__ inline ReduceBlockAux(ReduceOp *op, AscendC::GlobalTensor<InDtype> *input,
        AscendC::GlobalTensor<OutDtype> *output, const DataType *tiling)
    {
        this->op_ = op;
        this->input = input;
        this->output = output;
        this->tiling_ = tiling;
        for (uint64_t i = 0; i < DIM; i++) {
            iterAddr_[i].stride = tiling_->shape[i];
        }
    }

    __aicore__ inline void SetLoopRange()
    {
        int32_t blockId = AscendC::GetBlockIdx();
        if constexpr (IsBlockCutA<LoopInfo>()) {
            loopAStartIndex = blockId * tiling_->factorACntPerCore;
            loopAEndIndex = loopAStartIndex + tiling_->factorACntPerCore;
            if (unlikely(loopAEndIndex > tiling_->factorATotalCnt)) {
                loopAEndIndex = tiling_->factorATotalCnt;
            }
            int32_t aAxisIdx = LoopInfo->loopACount - 1;
            int32_t aAxis = LoopInfo->loopAAxis[aAxisIdx];
            loopAAxisStep = OpsUtils::CeilDiv(tiling_->shape[aAxis], tiling_->ubFactorA);
            if constexpr (LoopInfo->loopInnerRCount > 0) {
                constexpr int32_t rAxisIdx = LoopInfo->loopInnerRCount - 1;
                constexpr int32_t rAxis = LoopInfo->loopInnerRAxis[rAxisIdx];
                loopRAxisStep = OpsUtils::CeilDiv(tiling_->shape[rAxis], tiling_->ubFactorR);
            }
        } else {
            loopRStartIndex = blockId / tiling_->groupR * tiling_->factorRTotalCnt +
                               blockId % tiling_->groupR * tiling_->factorRCntPerCore;
            loopREndIndex = loopRStartIndex + tiling_->factorRCntPerCore;
            uint64_t maxRCnt = (blockId / tiling_->groupR + 1) * tiling_->factorRTotalCnt;
            uint64_t totalCnt = tiling_->factorATotalCnt * tiling_->factorRTotalCnt;
            maxRCnt = maxRCnt > totalCnt ? totalCnt : maxRCnt;
            if (unlikely(loopRStartIndex > maxRCnt)) {
                loopRStartIndex = maxRCnt;
            }
            if (unlikely(loopREndIndex > maxRCnt)) {
                loopREndIndex = maxRCnt;
            }

            int32_t rAxisIdx = LoopInfo->loopRCount - 1;
            int32_t rAxis = LoopInfo->loopRAxis[rAxisIdx];
            // The number of split axis Rfactors
            loopRAxisStep = OpsUtils::CeilDiv(tiling_->shape[rAxis], tiling_->ubFactorR);

            if constexpr (LoopInfo->loopACount > 0) {
                int32_t aAxisIdx = LoopInfo->loopACount - 1;
                int32_t aAxis = LoopInfo->loopAAxis[aAxisIdx];
                loopAAxisStep = OpsUtils::CeilDiv(tiling_->shape[aAxis], tiling_->ubFactorA);
            }
        }
        ubFactorA = tiling_->ubFactorA;
        ubFactorR = tiling_->ubFactorR;
    }

    template <class... Args>
    __aicore__ inline void Process(Args... args)
    {
        SetLoopRange();
        //Construct UB internal axis index array
        //1. Tail axis
        //2. Non cyclic axis
        //3. The innermost axis of the extranuclear A-cycle axis and UbFactorA>1
        //4. The innermost axis of the extranuclear R-cycle axis and UbFactorR>1
        //5. The innermost axis of the nuclear R-cycle axis and UbFactorR>1
        if constexpr (LoopInfo->loopRCount == 0) {
            rCount = tiling_->factorRCntPerCore;
        } else {
            rCount = loopREndIndex - loopRStartIndex;
        }
        bisectionPos = KernelUtils::FindNearestPower(rCount);
        cacheCount = KernelUtils::CalLog(bisectionPos) + 1;
        bisectionTail = rCount - bisectionPos;
        SetEvent<AscendC::HardEvent::V_MTE2>(AscendC::HardEvent::V_MTE2);
        if constexpr (LoopInfo->loopRCount == 0) {
            for (uint64_t i = loopAStartIndex; i < loopAEndIndex; i++) {
                CalcIterA<LoopInfo->loopACount>(i);
                IterateInnerA<0, LoopInfo->loopInnerACount>(args...);
            }
        } else {
            IterateInnerA<0, LoopInfo->loopInnerACount>(args...);
        }
    }

    template <int32_t start = 0, int32_t end = 0, class... Args>
    __aicore__ inline void IterateInnerA(Args... args)
    {
        if constexpr (start == end) {
            if constexpr (LoopInfo->reduceDichotomy) {
                int64_t tmpBufOffest = 0;
                Shape<InnerPattern::Dim> shape;
                if constexpr (LoopInfo->loopRCount == 0) {
                    LinearComputeR(tmpBufOffest, shape, args...);
                    CopyOut(tmpBufOffest, shape, args...);
                } else {
                    LinearComputeR(tmpBufOffest, shape, args...);
                    CopyOutGroup(tmpBufOffest, shape, args...);
                }
                SetEvent<AscendC::HardEvent::MTE3_V>(AscendC::HardEvent::MTE3_V);
            }
        } else {
            constexpr int32_t axis = LoopInfo->loopInnerAAxis[start];
            uint64_t shape = tiling_->shape[axis];
            if constexpr (start + 1 == end) {
                uint64_t loopSize = shape / this->ubFactorA;
                uint64_t tail = shape - loopSize * this->ubFactorA;
                this->iterAddr_[axis].start = 0;
                this->iterAddr_[axis].stride = this->ubFactorA;

                for (uint64_t i = 0; i < loopSize; i++) {
                    IterateInnerA<start + 1, end>(args...);
                    this->iterAddr_[axis].start += this->ubFactorA;
                }

                if (tail) {
                    this->iterAddr_[axis].stride = shape - this->iterAddr_[axis].start;
                    IterateInnerA<start + 1, end>(args...);
                }
            } else {
                for (uint64_t i = 0; i < shape; i++) {
                    this->iterAddr_[axis].start = i;
                    IterateInnerA<start + 1, end>(args...);
                }
            }
        }
    }

    template <bool isPadding, bool isTail, class V, class U>
    __aicore__ inline void PrePareReduce(int64_t index, V& view, U& shape, AscendC::LocalTensor<InDtype>& ubTensor,
                                      AscendC::LocalTensor<PromoteDataType>& computeTensor)
    {
        if constexpr (isTail) {
            index = index + bisectionPos;
        }
        if constexpr (LoopInfo->loopRCount > 0) {
            CalcIterR<LoopInfo->loopRCount>(index + loopRStartIndex);
        } else {
            CalcInnerIterR<LoopInfo->loopInnerRCount>(index);
        }
        CalcCopyInParam(view);
        if (index == 0) {
            CalcInnerShape(view, shape);
        }
        if constexpr (AscendC::IsSameType<PromoteDataType, InDtype>::value) {
            CopyIn<isPadding>(view, shape, ubTensor);
            SetEvent<AscendC::HardEvent::MTE2_V>(AscendC::HardEvent::MTE2_V);
            computeTensor = ubTensor;
        } else {
            // The index of AlloccomputeTensorAux does not require external perception
            op_->ReduceOp::template AllocTensorAux<false>(computeTensor);
            CopyIn<isPadding>(view, shape, ubTensor);
            SetEvent<AscendC::HardEvent::MTE2_V>(AscendC::HardEvent::MTE2_V);
            AscendC::Cast(computeTensor, ubTensor, AscendC::RoundMode::CAST_NONE, shape.value[0] * shape.value[1]);
            op_->FreeTensorAux(ubTensor);
        }
    }

    template <class V, class... Args>
    __aicore__ inline void LinearComputeR(int64_t& tmpBufOffest, V& shape, Args... args)
    {
        SliceView<MAX_DIM> view;
        for (int64_t i = 0; i < bisectionTail; i++) {
            AscendC::LocalTensor<InDtype> tensorLeft;
            op_->ReduceOp::template AllocTensorAux<true>(tensorLeft);
            AscendC::LocalTensor<PromoteDataType> computeLeft;
            PrePareReduce<(!InnerPattern::TailA), false>(i, view, shape, tensorLeft, computeLeft);
            
            AscendC::LocalTensor<InDtype> tensorRight;
            op_->ReduceOp::template AllocTensorAux<true>(tensorRight);
            AscendC::LocalTensor<PromoteDataType> computeRight;
            PrePareReduce<(!InnerPattern::TailA), true>(i, view, shape, tensorRight, computeRight);
            ComputeMerge(shape, computeLeft, computeRight, args...);

            // FP32 tensorLeft and computeLet are the same tensor, 
            // while FP16 computeLet does not require an index when free
            op_->ReduceOp::template FreeTensorAux(computeRight);
            op_->compute_.template UpdateCache<Pattern, V>(i, shape, op_->tempBuf_, op_->computeRes_);
        }

        for (int64_t i = bisectionTail; i < bisectionPos; i++) {
            AscendC::LocalTensor<InDtype> tensor;
            op_->ReduceOp::template AllocTensorAux<true>(tensor);
            AscendC::LocalTensor<PromoteDataType> computeLeft;
            PrePareReduce<(!InnerPattern::TailA && Pattern::Dim > 2), false>(i, view, shape, tensor, computeLeft);
            Compute(shape, computeLeft, args...);
            op_->ReduceOp::template FreeTensorAux(computeLeft);
            op_->compute_.template UpdateCache<Pattern>(i, shape, op_->tempBuf_, op_->computeRes_);
        }
        int64_t dimA = Pattern::TailA ? shape.value[1] : shape.value[0];
        int64_t cacheStride = OpsUtils::CeilDiv(dimA, static_cast<int64_t>(ELEMENT_ONE_REPEAT)) * ELEMENT_ONE_REPEAT;
        tmpBufOffest = (cacheCount - 1) * cacheStride;
    }

    template <int32_t LoopInnerRIdx>
    __aicore__ inline void CalcInnerIterR(uint64_t basicBlockIdx)
    {
        if constexpr (LoopInnerRIdx != 0) {
            constexpr auto axis = LoopInfo->loopInnerRAxis[LoopInnerRIdx - 1];
            if constexpr (LoopInnerRIdx == LoopInfo->loopInnerRCount) {
                // innermost loop
                auto cur = basicBlockIdx % this->loopRAxisStep;
                this->iterAddr_[axis].start = cur * this->ubFactorR;
                this->iterAddr_[axis].stride = tiling_->shape[axis] - this->iterAddr_[axis].start;
                if (likely(this->iterAddr_[axis].stride >= this->ubFactorR)) {
                    this->iterAddr_[axis].stride = this->ubFactorR;
                }
                CalcInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / this->loopRAxisStep);
            } else {
                this->iterAddr_[axis].start = basicBlockIdx % tiling_->shape[axis];
                this->iterAddr_[axis].stride = 1;
                CalcInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / tiling_->shape[axis]);
            }
        }
    }

    template <int32_t LoopAIdx>
    __aicore__ inline void CalcIterA(uint64_t step)
    {
        if constexpr (LoopAIdx != 0) {
            constexpr auto axis = LoopInfo->loopAAxis[LoopAIdx - 1];
            if constexpr (LoopAIdx == LoopInfo->loopACount) {
                // Splitting axis
                auto cur = step % this->loopAAxisStep;
                this->iterAddr_[axis].start = cur * this->ubFactorA;
                this->iterAddr_[axis].stride = tiling_->shape[axis] - this->iterAddr_[axis].start;
                if (likely(this->iterAddr_[axis].stride >= this->ubFactorA)) {
                    this->iterAddr_[axis].stride = this->ubFactorA;
                }

                if constexpr (LoopAIdx > 0) {
                    CalcIterA<LoopAIdx - 1>(step / this->loopAAxisStep);
                }
            } else {
                this->iterAddr_[axis].start = step % tiling_->shape[axis];
                this->iterAddr_[axis].stride = 1;
                CalcIterA<LoopAIdx - 1>(step / tiling_->shape[axis]);
            }
        }
    }

    template <int32_t LoopRIdx>
    __aicore__ inline void CalcIterR(uint64_t step)
    {
        uint64_t temp = step;
        if constexpr (LoopRIdx != 0) {
            for (auto idx = LoopInfo->loopRCount - 1; idx > -1; --idx) {
                if (idx == LoopInfo->loopRCount - 1) {
                    constexpr auto axis = LoopInfo->loopRAxis[LoopInfo->loopRCount - 1];
                    auto cur = temp % this->loopRAxisStep;
                    this->iterAddr_[axis].start = cur * this->ubFactorR;
                    this->iterAddr_[axis].stride = tiling_->shape[axis] - this->iterAddr_[axis].start;
                    if (likely(this->iterAddr_[axis].stride >= this->ubFactorR)) {
                        this->iterAddr_[axis].stride = this->ubFactorR;
                    }
                    temp = temp / this->loopRAxisStep;
                } else {
                    auto axis = LoopInfo->loopRAxis[idx];
                    if (IsLoopSpliteAAxis<LoopInfo>(axis)) {
                        auto cur = temp % this->loopAAxisStep;
                        this->iterAddr_[axis].start = cur * this->ubFactorA;
                        this->iterAddr_[axis].stride = tiling_->shape[axis] - this->iterAddr_[axis].start;
                        if (likely(this->iterAddr_[axis].stride >= this->ubFactorA)) {
                            this->iterAddr_[axis].stride = this->ubFactorA;
                        }
                        temp = temp / this->loopAAxisStep;
                    } else {
                        this->iterAddr_[axis].start = temp % tiling_->shape[axis];
                        this->iterAddr_[axis].stride = 1;
                        temp = temp / tiling_->shape[axis];
                    }
                }
            }
        }
    }

    template <class V>
    __aicore__ inline void CalcCopyInParam(V& view)
    {
        uint64_t addrOffset = 0;
        for (int32_t i = 0; i < DIM; i++) {
            addrOffset += iterAddr_[i].start * tiling_->stride[i];
        }
        constexpr static auto burstLenAxis = DIM - 1;  // Get the first loop axis
        view.addr = addrOffset;                        // Address to be moved
        view.burstLen = GetBurstLen<LoopInfo, burstLenAxis>(iterAddr_, tiling_);
        view.axisSize = 0;
        if constexpr (burstLenAxis > 0) {
            int32_t axis = burstLenAxis;
            for (int32_t i = 0; i < DIM; i++) {
                view.axisSize = i + 1;
                view.axis[i].repeat =
                    GetRepeatStride<LoopInfo>(axis - 1, iterAddr_, tiling_, view.axis[i].srcStride);
                view.axis[i].idx = GetRepeatStrideAxis<LoopInfo>(axis - 1, iterAddr_);
                view.axis[i].isAxisA = IsAxisA<Pattern::FirstA>(view.axis[i].idx);
                if (view.axis[i].idx <= 0) {
                    break;
                }
                axis = view.axis[i].idx;
            }
        }
    }

    template <class V, class U>
    __aicore__ inline void CalcInnerShape(V& view, U& shape)
    {
        shape.oriBurstLen = view.burstLen;
        if constexpr (!InnerPattern::TailA) {
            int64_t value = OpsUtils::CeilAlign(view.burstLen, BLOCK_SIZE_BYTE / sizeof(InDtype));
            if (IsLoopSpliteRAxis<LoopInfo>(DIM - 1)) {
                value = OpsUtils::CeilAlign(this->ubFactorR, BLOCK_SIZE_BYTE / sizeof(InDtype));
            }
            for (uint64_t i = 0; i < view.axisSize; i++) {
                if (!view.axis[i].isAxisA) {
                    view.axis[i].dstStride = value;
                    if (IsLoopSpliteRAxis<LoopInfo>(view.axis[i].idx)) {
                        value = value * this->ubFactorR;
                    } else {
                        value = value * view.axis[i].repeat;
                    }
                }
            }
            shape.value[InnerPattern::Dim - 1] = value;
            for (uint64_t i = 0; i < view.axisSize; i++) {
                if (view.axis[i].isAxisA) {
                    view.axis[i].dstStride = value;
                    value = value * view.axis[i].repeat;
                }
            }
            shape.value[InnerPattern::Dim - 2] = value / shape.value[InnerPattern::Dim - 1];
        } else {
            int64_t value = OpsUtils::CeilAlign(view.burstLen, BLOCK_SIZE_BYTE / sizeof(InDtype));
            aOutBurstLen = view.burstLen;
            if (IsLoopSpliteRAxis<LoopInfo>(DIM - 1)) {
                value = OpsUtils::CeilAlign(this->ubFactorA, BLOCK_SIZE_BYTE / sizeof(InDtype));
                aOutBurstLen = this->ubFactorA;
            }
            aOutNBurst = 1;
            for (uint64_t i = 0; i < view.axisSize; i++) {
                if (view.axis[i].isAxisA) {
                    view.axis[i].dstStride = value;
                    value = value * view.axis[i].repeat;
                    aOutNBurst = aOutNBurst * view.axis[i].repeat;
                }
            }
            shape.value[InnerPattern::Dim - 1] = value;
            for (uint64_t i = 0; i < view.axisSize; i++) {
                if (!view.axis[i].isAxisA) {
                    view.axis[i].dstStride = value;
                    value = value * view.axis[i].repeat;
                }
            }
            shape.value[InnerPattern::Dim - 2] = value / shape.value[InnerPattern::Dim - 1];
        }
    }

    template <bool isPadding, class U, class V>
    __aicore__ inline void CopyIn(U& view, V& shape, const AscendC::LocalTensor<InDtype>& ubTensor)
    {
        // Calculate the View in UB
        op_->ReduceOp::template CopyInAux<isPadding, BLOCK_PRE_COMPUTE>(this->input[0], view, shape, ubTensor);
    }

    template <class V, class... Args>
    __aicore__ inline void Compute(V& shape, const AscendC::LocalTensor<PromoteDataType>& ubTensor, Args... args)
    {
        op_->ReduceOp::template ComputeAux<(!InnerPattern::TailA && Pattern::Dim <= 2), InnerPattern>(
            shape, ubTensor, args...);
    }

    template <class V, class... Args>
    __aicore__ inline void ComputeMerge(V& shape, const AscendC::LocalTensor<PromoteDataType>& ubTensorLeft,
                                        const AscendC::LocalTensor<PromoteDataType>& ubTensorRight, Args... args)
    {
        // Reduce between UB
        op_->compute_.ReduceBetweenUB(ubTensorLeft, ubTensorRight, shape.value[0] * shape.value[1]);
        op_->ReduceOp::template FreeTensorAux(ubTensorLeft);
        op_->ReduceOp::template ComputeAux<false, InnerPattern>(shape, ubTensorRight, args...);
        op_->ReduceOp::template FreeTensorAux(ubTensorRight);
    }

    template <class V>
    __aicore__ inline void CopyOut(int64_t tmpBufOffest, V& shape)
    {
        constexpr int32_t axis = Pattern::FirstA ? 0 : 1;
        uint64_t addrOffset = 0;
        for (int32_t i = axis; i < DIM; i += CONST2) {
            addrOffset += this->iterAddr_[i].start * tiling_->dstStride[i];
        }
        SliceView<CONST2> view;
        view.addr = addrOffset;
        if constexpr (Pattern::TailA) {
            view.burstLen = aOutBurstLen;
            view.axis[0].repeat = aOutNBurst;
        } else {
            view.burstLen = shape.value[0];
            view.axis[0].repeat = 1;
        }
        op_->ReduceOp::template CopyOutAux<BLOCK_POST_COMPUTE>(this->output[0], view, tmpBufOffest);
    }

    template <class V>
    __aicore__ inline void CopyOutGroup(int64_t tmpBufOffest, V& shape)
    {
        // CopyOut As RA Pattern
        SliceView<CONST2> view;
        int32_t blockId = AscendC::GetBlockIdx();

        int32_t innerA = GetInnerA<LoopInfo, Pattern::TailA, Pattern::Dim>(iterAddr_);
        if constexpr (Pattern::TailA) {
            view.burstLen = aOutBurstLen;
            view.axis[0].repeat = aOutNBurst;
            view.axis[0].srcStride = OpsUtils::CeilAlign(aOutBurstLen, BLOCK_SIZE_BYTE / sizeof(InDtype));
        } else {
            view.burstLen = shape.value[0];
            view.axis[0].repeat = 1;
            view.axis[0].srcStride = shape.value[0];
        }
        int32_t axis = Pattern::FirstA ? 0 : 1;
        if constexpr (LoopInfo->loopACount > 0) {
            axis = LoopInfo->loopAAxis[LoopInfo->loopACount - 1];
        }

        uint64_t addrOffset = 0;
        if constexpr (LoopInfo->loopInnerACount > 0) {
            for (int32_t i = axis; i < DIM; i += CONST2) {
                addrOffset += this->iterAddr_[i].start * tiling_->dstStride[i];
            }
        }

        uint64_t axisStep = LoopInfo->loopACount > 0 ? this->loopAAxisStep : 1;
        view.addr = (blockId % tiling_->groupR) * tiling_->outSize +                            // group offset
                    (blockId / (tiling_->groupR * axisStep)) * tiling_->shape[axis] * innerA +  // AAxis offset
                    (blockId / tiling_->groupR % axisStep) * this->ubFactorA * innerA +        // main offset
                    addrOffset;                                                                 // innerA offset

        op_->ReduceOp::template CopyOutAuxGroup(this->output[0], view, tmpBufOffest);
    }
};
}  // namespace Reduce
}  // namespace KernelUtils
}  // namespace ATVC
#endif // ATVC_REDUCE_BLOCK_AUX_H
