/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_PRE_COMPUTE_ADD_OF_REDUCE_SUM_H
#define ATVC_PRE_COMPUTE_ADD_OF_REDUCE_SUM_H
#include "reduce/kernel/reduce_kernel.h"

template <typename Traits>
struct PreComputeAddOfReduceSum {
    using Inputs = typename Traits::In::types;
    using DataType = typename ATVC::TypeListGet<Inputs, 0>::Type;

    /* !
     * \brief set scaler param for compute function
     * \param [in] args, args are mutable parameters, and are passed transparently from the parameters of
     *                   global kernel functions, which are the parameters after reduceParam
     */
    template <class... Args>
    __aicore__ inline void SetScalar(Args... args)
    {}

    /* !
     * \brief set tensor param for compute function
     * \param [in] args, args are mutable parameters, and are passed transparently from the parameters of
     *                   global kernel functions, which are the parameters before reduceParam, the num of args is
     *                   decided by Traits
     */
    template <class... Args>
    __aicore__ inline void SetArgs(Args... args)
    {
        InitArgsInput<0>(args...);
    }

    /* !
     * \brief process function of compute struct
     * \param [in] x, local tensor of x
     * \param [in] reduceIn, local tensor of reduceIn,  reduceIn is the input of reduce_sum, must be the last local
     * tensor \param [in] copyInOffset, copy In offset for DataCopy \param [in] copyInParams, copy In params for
     * DataCopy \param [in] padParams, copy In padding for DataCopy
     */
    template <typename DataType>
    __aicore__ inline void operator()(AscendC::LocalTensor<DataType> x,
                                        AscendC::LocalTensor<DataType> reduceIn,
                                        uint32_t dstOffset,
                                        uint32_t copyInOffset,
                                        AscendC::DataCopyExtParams &copyInParams,
                                        AscendC::DataCopyPadExtParams<DataType> padParams)
    {
        int32_t dataUnit = sizeof(DataType);
        ATVC::SyncDataQueue<AscendC::HardEvent::MTE3_MTE2>();
        CopyIn<DataType>(x[dstOffset], copyInOffset, copyInParams, padParams);
        AscendC::PipeBarrier<PIPE_V>();  // wait reduce finished
        ATVC::SyncDataQueue<AscendC::HardEvent::MTE2_V>();

        int32_t pad = padParams.isPad ? 0 : padParams.leftPadding + padParams.rightPadding;
        int32_t dataLen = copyInParams.blockLen + dataUnit * pad;
        int32_t dataAlignLen = OpsUtils::CeilDiv(dataLen, ATVC::UB_ALIGN_32) * ATVC::UB_ALIGN_32;
        int32_t stride = copyInParams.dstStride * ATVC::UB_ALIGN_32;
        int32_t repStride = OpsUtils::CeilDiv((dataAlignLen + stride), ATVC::UB_ALIGN_32);
        int32_t offset = 0;

        if (repStride > ATVC::UB_ALIGN_255) {
            // Calculate all data when the stride exceeds the max-value(255)
            for (int32_t i = 0; i < copyInParams.blockCount; i++) {
                offset = i * (dataAlignLen + stride) / dataUnit;
                AscendC::Adds(reduceIn[dstOffset + offset],
                    x[dstOffset + offset],
                    DataType{1},
                    copyInParams.blockLen / dataUnit);
            }
        } else {
            Compute(x, reduceIn, dstOffset, copyInParams, padParams);
        }
    }

private:
    template <int32_t start, class... Args>
    __aicore__ inline void InitArgsInput(GM_ADDR src, Args... args)
    {
        srcGlobal_.SetGlobalBuffer((__gm__ DataType *)src);
    }

    template <typename DataType>
    __aicore__ inline void CopyIn(AscendC::LocalTensor<DataType> x, uint32_t copyInOffset,
        AscendC::DataCopyExtParams &copyInParams, AscendC::DataCopyPadExtParams<DataType> padParams)
    {
        AscendC::DataCopyPad(x, srcGlobal_[copyInOffset], copyInParams, padParams);
    }
    template <typename DataType>
    __aicore__ inline void Compute(AscendC::LocalTensor<DataType> x, AscendC::LocalTensor<DataType> reduceIn,
        uint32_t dstOffset, AscendC::DataCopyExtParams &copyInParams, AscendC::DataCopyPadExtParams<DataType> padParams)
    {
        int32_t dataUnit = sizeof(DataType);
        int32_t pad = padParams.isPad ? 0 : padParams.leftPadding + padParams.rightPadding;
        int32_t dataLen = copyInParams.blockLen + dataUnit * pad;
        int32_t dataAlignLen = OpsUtils::CeilDiv(dataLen, ATVC::UB_ALIGN_32) * ATVC::UB_ALIGN_32;
        int32_t stride = copyInParams.dstStride * ATVC::UB_ALIGN_32;
        int32_t repStride = OpsUtils::CeilDiv((dataAlignLen + stride), ATVC::UB_ALIGN_32);
        int32_t offset = 0;
        // Calculate data when the stride does not exceeds the max-value(255)
        int32_t repeatStrideCycle = dataLen / ATVC::UB_ALIGN_256;
        int32_t maxMaskLen = ATVC::UB_ALIGN_256 / dataUnit;
        int32_t repeatTime = copyInParams.blockCount;
        AscendC::UnaryRepeatParams repeatParams;
        repeatParams.dstRepStride = repStride;
        repeatParams.srcRepStride = repStride;
        int32_t repeatTimeCycle = copyInParams.blockCount / ATVC::UB_ALIGN_255;
        int32_t dataTail = dataLen - repeatStrideCycle * ATVC::UB_ALIGN_256;
        int32_t dataAlignTail = dataAlignLen - repeatStrideCycle * ATVC::UB_ALIGN_256;
        int32_t mask = dataTail / dataUnit;

        // Calculate the entire block of data for repeatTime
        for (int32_t i = 0; i < repeatTimeCycle; i++) {
            offset = (dataAlignLen + stride) / dataUnit * ATVC::UB_ALIGN_255 * i;
            for (int32_t j = 0; j < repeatStrideCycle; j++) {
                AscendC::Adds(reduceIn[dstOffset + j * maxMaskLen + offset],
                    x[dstOffset + j * maxMaskLen + offset],
                    DataType{1}, maxMaskLen, ATVC::UB_ALIGN_255, repeatParams);
            }
            // Calculate the entire block of data for repeatStride
            if (dataTail > 0) {
                AscendC::Adds(reduceIn[dstOffset + repeatStrideCycle * maxMaskLen + offset],
                    x[dstOffset + repeatStrideCycle * maxMaskLen + offset],
                    DataType{1}, mask, ATVC::UB_ALIGN_255, repeatParams);
            }
        }

        offset = (dataAlignTail + stride) / dataUnit * repeatTimeCycle * ATVC::UB_ALIGN_255;
        int32_t repeatTimeTail = copyInParams.blockCount - ATVC::UB_ALIGN_255 * repeatTimeCycle;
        // Calculate the tail block of data for repeatTime
        if (repeatTimeTail > 0) {
            for (int32_t i = 0; i < repeatStrideCycle; i++) {
                AscendC::Adds(reduceIn[dstOffset + offset + i * maxMaskLen], x[dstOffset + offset + i * maxMaskLen],
                    DataType{1}, maxMaskLen, repeatTimeTail, repeatParams);
            }
            // Calculate the tail block of data for repeatStride
            offset = (dataAlignTail + stride) / dataUnit * repeatTimeCycle * ATVC::UB_ALIGN_255 + repeatStrideCycle * maxMaskLen;
            if (dataTail > 0) {
                AscendC::Adds(reduceIn[dstOffset + offset], x[dstOffset + offset],
                    DataType{1}, mask, repeatTimeTail, repeatParams);
            }
        }
    }

    AscendC::GlobalTensor<DataType> srcGlobal_;
};

#endif
