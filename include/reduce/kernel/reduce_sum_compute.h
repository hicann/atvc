/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_REDUCE_SUM_H
#define ATVC_REDUCE_SUM_H

#include "common/kernel_utils.h"
#include "reduce/common/patterns.h"
#include "reduce/kernel/utils/reduce_block_aux_util.h"

namespace {
struct ReduceARParam {
    uint32_t repStride = 0;
    uint16_t dimA = 0;
    uint16_t dimMax = 0;
    uint16_t mainR = 0;
    uint16_t tailR = 0;
    uint64_t maskAddRNum = 0;
    uint16_t loopRNum = 0;
    uint16_t dtypeSize = 0;
    uint16_t dimR = 0;
};
}

namespace ATVC {
/*!
 * ReduceSumCompute: This class provides the core arithmetic required to reduce
 * tensors along either the inner-most (AR) or outer-most (RA) axis after
 * the tensor has been copied to the Unified Buffer (UB).  Data movement between
 * Global Memory (GM) and UB is not handled here; it is the responsibility of
 * the surrounding scheduling template.
 */
template <typename OpTraits>
class ReduceSumCompute {
public:
    // Extract operator input description information from OpTraits
    using inputDTypeList = typename OpTraits::In::types;
    using DataType = typename ATVC::TypeListGet<inputDTypeList, 0>::Type;
    using PrompteDtype = typename KernelUtils::GetPromoteType<DataType>::T;
    __aicore__ inline ReduceSumCompute() {}

    /*!
     * \brief Perform the actual reduction on a tile already resident in UB.
     * \tparam needMask, true when UB alignment introduced invalid lanes.
     * \tparam Pattern, one of ReducePattern::AR or ReducePattern::RA.
     * \param[in] shape, {dimA, dimR} in elements; dimR may be padded.
     * \param[out] dst, destination tensor (length == dimA)
     * \param[in] src, source tensor (length == dimA * dimR)
     */
    template <bool needMask, class Pattern>
    __aicore__ inline void
    Compute(KernelUtils::Shape<2> &shape,
            const AscendC::LocalTensor<PrompteDtype> &dst,
            const AscendC::LocalTensor<PrompteDtype> &src)
    {
        // AR scenario, hardware limitations, R-axis requires 32B alignment on UB, with 2 alignment methods available:
        // 1. High performance alignment (with uncertain supplementary element values), subsequent cumulative
        //    calculations can only calculate the actual number of effective elements
        // 2. Alignment with zero padding (padding value is determined by the GetAddingValue() interface
        //    implemented by the user)
        if (std::is_same<Pattern, ReducePattern::AR>::value) {
            if constexpr (needMask) { // 1. High performance alignment mode
                // MainR (int64_t dimR, boolean isAR): The framework provides the calculation of the R-axis binary
                // length (number of elements), where dimR is the original number of elements
                int16_t mainR = KernelUtils::Reduce::MainR(shape.oriBurstLen, true);
                ReduceAR(dst, src, shape.value[0], shape.value[1],  mainR, shape.oriBurstLen);
            } else {
                // MainR: The framework provides the calculation of the R-axis binary length (number of elements),
                // where dimR is the number of elements after completion
                int16_t mainR = KernelUtils::Reduce::MainR(shape.value[1], true);
                ReduceAR(dst, src, shape.value[0], shape.value[1],  mainR, shape.value[1]);
            }
        }
        if (std::is_same<Pattern, ReducePattern::RA>::value) {
            int16_t mainR = KernelUtils::Reduce::MainR(shape.value[0], false);
            ReduceRA(dst, src, shape.value[1], shape.value[0], mainR);
        }
    }

    /*!
     * \brief RA-pattern reduction: reduce along the outer-most (slowest-varying) axis.
     * \param[out] dst, output tensor (length == dimA)
     * \param[in] src, input tensor (length == dimR * dimA), already resident in UB
     * \param[in] dimA, length of the non-reduced axis (A)
     * \param[in] dimR, length of the reduced axis (R)
     * \param[in] mainR, largest power-of-two ≤ dimR (computed by the caller)
     */
    __aicore__ inline void
    ReduceRA(const AscendC::LocalTensor<PrompteDtype> &dst,
             const AscendC::LocalTensor<PrompteDtype> &src, uint16_t dimA,
             uint16_t dimR, uint16_t mainR)
    {
        uint32_t totalNum = dimR * dimA;
        uint32_t mainNum = dimA * mainR;
        constexpr uint32_t dtypeSize = sizeof(PrompteDtype);
        uint32_t tailNum = totalNum - mainNum;
        // MaskAddNum has a maximum value of 256 bytes and must be aligned with 32 bytes
        constexpr uint32_t maskAddNum = UB_ALIGN_256 / dtypeSize / UB_ALIGN_32 * UB_ALIGN_32;
        uint16_t repeatTimes = tailNum / maskAddNum;
        uint16_t repeatNum = repeatTimes * maskAddNum;
        uint16_t repTailNum = tailNum - repeatNum;
        // Same data block step size between different iterations
        uint32_t repStride = dtypeSize * maskAddNum / UB_ALIGN_32;
        // dstBlkStride, src0BlkStride,src1BlkStride, dstRepStride, src0RepStride, src1RepStride
        AscendC::BinaryRepeatParams repeatParams(1, 1, 1, repStride, repStride, repStride);
        if (repeatTimes > 0) {
            AscendC::Add(src, src[mainNum], src, maskAddNum, repeatTimes, repeatParams);
        }
        if (repTailNum > 0) {
            // Same data block step size between different iterations
            repStride = dtypeSize * repTailNum / UB_ALIGN_32;
            repeatParams.dstRepStride = repStride;
            repeatParams.src0RepStride = repStride;
            repeatParams.src1RepStride = repStride;
            AscendC::Add(src[repeatNum], src[repeatNum + mainNum], src[repeatNum], repTailNum, 1, repeatParams);
        }
        AscendC::PipeBarrier<PIPE_V>();
        uint16_t loopRNum = mainR;
        while (loopRNum > 1) {
            loopRNum = loopRNum >> 1;
            mainNum = loopRNum * dimA; // The first half of LoopR's data volume
            repeatTimes = mainNum / maskAddNum;
            repeatNum = repeatTimes * maskAddNum;
            repTailNum = mainNum - repeatNum;
            if (repeatTimes > 0) {
                // Same data block step size between different iterations
                repStride = dtypeSize * maskAddNum / UB_ALIGN_32;
                repeatParams.dstRepStride = repStride;
                repeatParams.src0RepStride = repStride;
                repeatParams.src1RepStride = repStride;
                AscendC::Add(src, src[mainNum], src, maskAddNum, repeatTimes, repeatParams);
            }
            if (repTailNum > 0) {
                // Same data block step size between different iterations
                repStride = dtypeSize * repTailNum / UB_ALIGN_32;
                repeatParams.dstRepStride = repStride;
                repeatParams.src0RepStride = repStride;
                repeatParams.src1RepStride = repStride;
                AscendC::Add(src[repeatNum], src[repeatNum + mainNum], src[repeatNum], repTailNum, 1, repeatParams);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::DataCopy(dst, src, dimA);
    }

    /*!
     * \brief AR-pattern reduction: reduce along the inner-most (fastest-varying) axis.
     * \param[out] dstTensor, output tensor (length == dimA)
     * \param[in] srcTensor, input tensor (length == dimR * dimA), already resident in UB
     * \param[in] dimA, length of the non-reduced axis (A)
     * \param[in] dimR, padded length of the reduced axis (R)
     * \param[in] mainR, largest power-of-two ≤ original R length
     * \param[in] oriBurstLen, original (un-padded) R length used to compute tail
     */
    __aicore__ inline void
    ReduceAR(const AscendC::LocalTensor<PrompteDtype> &dstTensor,
             const AscendC::LocalTensor<PrompteDtype> &srcTensor, uint16_t dimA,
             uint16_t dimR, uint16_t mainR, uint64_t oriBurstLen)
    {
        uint16_t tailR = oriBurstLen - mainR;
        constexpr uint16_t dtypeSize = sizeof(PrompteDtype);
        uint32_t repStride = dtypeSize * dimR / UB_ALIGN_32;
        uint16_t dimMax = dimA * dimR;
        constexpr uint64_t maskAddRNum = UB_ALIGN_256 / dtypeSize;

        ReduceARParam param{
            .repStride   = repStride,
            .dimA        = dimA,
            .dimMax      = dimMax,
            .mainR       = mainR,
            .tailR       = tailR,
            .maskAddRNum = maskAddRNum,
            .dtypeSize   = dtypeSize,
            .dimR        = dimR
        };

        if (mainR > 0 && tailR > 0) {
            PerformInitialAdd(srcTensor, param);
        }

        param.loopRNum = mainR;
        while (param.loopRNum > maskAddRNum) {
            param.loopRNum = param.loopRNum / 2U;
            PerformBinaryReduction(srcTensor, param);
        }
        if (param.loopRNum == 0) { // small shape, directly reduce
            param.loopRNum = tailR;
        }
        PerformFinalReduction(dstTensor, srcTensor, param);
    }

    /*!
     * \brief Merge the calculation results of different data base blocks within a single UB
     * \tparam Pattern  Compile-time pattern tag that decides A vs. B orientation.
     * \tparam V Shape descriptor (encodes dimA and dimB at runtime).
     * \param[in] index, logical index identifying the data-base block.
     * \param[in] shape, runtime tensor shape (dimA, dimB).
     * \param[in] tempBuf, UB tensor serving as the reduction cache.
     * \param[in] computeRes, UB tensor holding the newest partial result.
     */
    template <class Pattern, class V>
    __aicore__ inline void UpdateCache(int64_t index, V& shape, const AscendC::LocalTensor<PrompteDtype>& tempBuf,
                                       const AscendC::LocalTensor<PrompteDtype>& computeRes)
    {
        int64_t cacheID = KernelUtils::Reduce::GetCacheID(index);
        int64_t dimA = Pattern::TailA ? shape.value[1] : shape.value[0];
        int32_t elementOneRepeat = Platform::GetVRegSize() / sizeof(PrompteDtype);
        int64_t stride = OpsUtils::CeilDiv(dimA, static_cast<int64_t>(elementOneRepeat)) * elementOneRepeat;
        uint16_t outerLoopTimes = OpsUtils::CeilDiv(
            static_cast<int64_t>(dimA * sizeof(PrompteDtype)), static_cast<int64_t>(Platform::GetVRegSize()));
        uint16_t innerLoopTimes = cacheID;
        uint32_t outerLoopStride = elementOneRepeat;
        uint32_t innerLoopStride = stride;  // The size of each idex block in cacahe and the size of the A-axis
        AscendC::LocalTensor<PrompteDtype> dstTensor = tempBuf;
        AscendC::LocalTensor<PrompteDtype> srcTensor = computeRes;
        uint32_t cah = cacheID * stride;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) {  // OuterLoopTimes is the size of dimA
            uint32_t srcIdx = i * outerLoopStride;
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                AscendC::Add(srcTensor[srcIdx], srcTensor[srcIdx],
                             dstTensor[srcIdx + j * innerLoopStride],
                             outerLoopStride);
                AscendC::PipeBarrier<PIPE_V>();
            }
            DataCopy(dstTensor[cah + srcIdx], srcTensor[srcIdx], outerLoopStride);
        }
    }

    /*!
     * \brief Binary reduction between two UB buffers.
     * \      Used for inter-core result merging when workspace staging is required.
     * \param[in] ubTensorLeft, left operand (in-place result).
     * \param[in] ubTensorRight, right operand (read-only).
     * \param[in] calCount, number of elements to reduce.
     */
    __aicore__ inline void
    ReduceBetweenUB(const AscendC::LocalTensor<PrompteDtype> &ubTensorLeft,
                    const AscendC::LocalTensor<PrompteDtype> &ubTensorRight,
                    const int32_t &calCount)
    {
        Add(ubTensorRight, ubTensorRight, ubTensorLeft, calCount);
    }

    /*!
     * \brief Return the value used for padding when UB alignment is required.
     *        For SUM-reduction the neutral element is 0.
     * \tparam U, scalar type identical to DataType or PromoteDataType.
     * \return The padding value (always 0).
     */
    template <typename U>
    __aicore__ inline U GetPaddingValue()
    {
        // Due to the fact that ReduceSum accumulates R-axis data, the values of the supplemented elements
        // are set to 0 to ensure that the accumulated result is not affected
        U paddingValue = 0;
        return paddingValue;
    }

private:
    __aicore__ inline void PerformInitialAdd(const AscendC::LocalTensor<PrompteDtype> &srcTensor, const ReduceARParam& param)
    {
        uint16_t addRTotalNum = param.tailR / param.maskAddRNum * param.maskAddRNum;
        uint16_t addRTail = param.tailR - addRTotalNum;
        // dstBlkStride, src0BlkStride,src1BlkStride, dstRepStride, src0RepStride, src1RepStride
        AscendC::BinaryRepeatParams repeatParams(1, 1, 1, param.repStride, param.repStride, param.repStride);

        if (param.repStride > UB_ALIGN_255) {
            for (uint16_t i = 0; i < param.dimMax; i += param.dimR) {
                AscendC::Add(srcTensor[i], srcTensor[i], srcTensor[i + param.mainR], param.tailR);
            }
        } else {
            for (uint16_t i = 0; i < addRTotalNum; i += param.maskAddRNum) {
                AscendC::Add(srcTensor[i], srcTensor[i + param.mainR], srcTensor[i], param.maskAddRNum, param.dimA, repeatParams);
            }
            if (addRTail > 0) {
                AscendC::Add(srcTensor[addRTotalNum],
                    srcTensor[addRTotalNum + param.mainR],
                    srcTensor[addRTotalNum],
                    addRTail,
                    param.dimA,
                    repeatParams);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PerformBinaryReduction(const AscendC::LocalTensor<PrompteDtype> &srcTensor,
        const ReduceARParam& param)
    {
        if (param.repStride > UB_ALIGN_255) {
            for (uint16_t i = 0; i < param.dimMax; i += param.loopRNum) {
                AscendC::Add(srcTensor[i], srcTensor[i], srcTensor[i + param.loopRNum], param.loopRNum);
            }
        } else {
            uint16_t addRTotalNum = param.loopRNum / param.maskAddRNum * param.maskAddRNum;
            uint16_t addRTail = param.loopRNum - addRTotalNum;
            // dstBlkStride, src0BlkStride,src1BlkStride, dstRepStride, src0RepStride, src1RepStride
            AscendC::BinaryRepeatParams repeatParams(1, 1, 1, param.repStride, param.repStride, param.repStride);
            for (uint16_t i = 0; i < addRTotalNum; i += param.maskAddRNum) {
                AscendC::Add(srcTensor[i], srcTensor[i + param.loopRNum], srcTensor[i], param.maskAddRNum, param.dimA, repeatParams);
            }
            if (addRTail > 0) {
                AscendC::Add(srcTensor[addRTotalNum],
                    srcTensor[addRTotalNum],
                    srcTensor[addRTotalNum + param.loopRNum],
                    addRTail,
                    param.dimA,
                    repeatParams);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PerformFinalReduction(const AscendC::LocalTensor<PrompteDtype> &dstTensor,
        const AscendC::LocalTensor<PrompteDtype> &srcTensor, const ReduceARParam& param)
    {
        if constexpr (AscendC::IsSameType<PrompteDtype, float>::value ||
                      AscendC::IsSameType<PrompteDtype, half>::value) {
            uint16_t reduceLoopTimes = UB_ALIGN_255 * param.dtypeSize / UB_ALIGN_32 * UB_ALIGN_32 / param.dtypeSize;
            // WholeReduceSum repeat-time limit is 255; split dimA into chunks
            for (uint16_t dimAIdx = 0; dimAIdx < param.dimA; dimAIdx += reduceLoopTimes) {
                uint16_t curDimA = (dimAIdx + reduceLoopTimes < param.dimA) ? reduceLoopTimes : param.dimA - dimAIdx;
                AscendC::WholeReduceSum(
                    dstTensor[dimAIdx], srcTensor[dimAIdx * param.dimR], param.loopRNum, curDimA, 1, 1, param.repStride);
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else if constexpr (AscendC::IsSameType<PrompteDtype, int32_t>::value ||
                             AscendC::IsSameType<PrompteDtype, uint32_t>::value) {
            // Cast to float for higher-precision accumulation
            AscendC::LocalTensor<float> interpreSrc = srcTensor.template ReinterpretCast<float>();
            AscendC::LocalTensor<float> interpreDst = dstTensor.template ReinterpretCast<float>();
            AscendC::Cast(interpreSrc, srcTensor, AscendC::RoundMode::CAST_NONE, param.dimA * param.dimR);
            AscendC::PipeBarrier<PIPE_V>();

            uint16_t reduceLoopTimes = 255 * param.dtypeSize / UB_ALIGN_32 * UB_ALIGN_32 / param.dtypeSize;
            // WholeReduceSum repeat-time limit is 255; split dimA into chunks
            for (uint16_t dimAIdx = 0; dimAIdx < param.dimA; dimAIdx += reduceLoopTimes) {
                uint16_t curDimA = (dimAIdx + reduceLoopTimes < param.dimA) ? reduceLoopTimes : param.dimA - dimAIdx;
                AscendC::WholeReduceSum(
                    interpreDst[dimAIdx], interpreSrc[dimAIdx * param.dimR], param.loopRNum, curDimA, 1, 1, param.repStride);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(dstTensor, interpreDst, AscendC::RoundMode::CAST_RINT, dstTensor.GetSize());
        }
    }
};
} // namespace ATVC

#endif // ATVC_REDUCE_SUM_H
