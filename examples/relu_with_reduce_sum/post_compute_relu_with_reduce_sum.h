/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_POST_COMPUTE_RELU_OF_REDUCE_SUM_H
#define ATVC_POST_COMPUTE_RELU_OF_REDUCE_SUM_H
#include "reduce/kernel/reduce_kernel.h"

template <typename Traits>
struct PostComputeReluOfReduceSum {
    using inputDTypeList = typename Traits::In::types;
    using DataType = typename ATVC::TypeListGet<inputDTypeList, 0>::Type;
    
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
        InitArgsOutput<0>(args...);
    }

    /* !
     * \brief process function of compute struct
     * \param [in] y, local tensor of y
     * \param [in] reduceOut, local tensor of reduceOut
     * \param [in] copyOutOffset, copy In offset for DataCopy
     * \param [in] copyOutParams, copy out params for DataCopy
     */
    template <typename DataType>
    __aicore__ inline void operator()(AscendC::LocalTensor<DataType> y, AscendC::LocalTensor<DataType> reduceOut,
        uint32_t copyOutOffset, AscendC::DataCopyExtParams &copyOutParams)
    {
        int32_t dataLen = copyOutParams.blockLen;
        int32_t dataAlignLen = OpsUtils::CeilDiv(dataLen, ATVC::UB_ALIGN_32) * ATVC::UB_ALIGN_32;
        size_t size =
                copyOutParams.blockCount * (dataAlignLen + copyOutParams.srcStride * ATVC::UB_ALIGN_32) / sizeof(DataType);
        ATVC::SyncDataQueue<AscendC::HardEvent::MTE2_V>();
        Compute<DataType>(y, reduceOut, size);
        ATVC::SyncDataQueue<AscendC::HardEvent::V_MTE3>();
        CopyOut<DataType>(y, copyOutOffset, copyOutParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    template <int32_t start, class... Args>
    __aicore__ inline void InitArgsOutput(GM_ADDR dst, Args... args)
    {
        dstGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(dst));
    }

    template <typename DataType>
    __aicore__ inline void Compute(
        AscendC::LocalTensor<DataType> y, AscendC::LocalTensor<DataType> reduceOut, uint32_t size)
    {
        AscendC::Relu(y, reduceOut, size);
    }

    template <typename DataType>
    __aicore__ inline void CopyOut(
        AscendC::LocalTensor<DataType> y, uint32_t copyOutOffset, AscendC::DataCopyExtParams &copyOutParams)
    {
        AscendC::DataCopyPad(dstGlobal_[copyOutOffset], y, copyOutParams);
    }
    
    AscendC::GlobalTensor<DataType> dstGlobal_;
};
#endif
