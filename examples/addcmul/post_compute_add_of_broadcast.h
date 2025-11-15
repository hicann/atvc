/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_POST_COMPUTE_ADD_OF_BROADCAST_H
#define ATVC_POST_COMPUTE_ADD_OF_BROADCAST_H

#include "broadcast/kernel/broadcast_kernel.h"

template<typename Traits>
struct PostComputeAddOfBroadcast {    
    using inputDTypeList = typename Traits::In::types;
    using DataType = typename ATVC::TypeListGet<inputDTypeList, 0>::Type;
    /* !
    * \brief set scaler param for compute fuction
    * \param [in] args, args are mutable parameters, and are passed transparently from the parameters of
    *                   global kernel functions, which are the parameters after broadcastParam
    */
    template <class... Args>
    __aicore__ inline void SetParam(Args... args) {}

    /* !
    * \brief set tensor param for compute fuction
    * \param [in] args, args are mutable parameters, and are passed transparently from the parameters of
    *                   global kernel functions, which are the parameters before broadcastParam, the num of args is
    *                   decided by Traits
    */
    template<class... Args>
    __aicore__ inline void SetArgs(Args... args)
    {
        InitArgs(args...);
    }

    /* !
    * \brief process function of compute struct
    * \param [in] y, local tensor of y
    * \param [in] z, local tensor of z
    * \param [in] x, local tensor of x,  x is the output of broadcast, must be the last local tensor
    * \param [in] copyOutOffset,  copy out offset for DataCopy
    * \param [in] copyOutParams, copy out params for DataCopy
    */
    template<typename DataType>
    __aicore__ inline void operator()(AscendC::LocalTensor<DataType> y, AscendC::LocalTensor<DataType> z, AscendC::LocalTensor<DataType> x,
        uint32_t copyOutOffset,  AscendC::DataCopyExtParams &copyOutParams)
    {
        size_t size = copyOutParams.blockCount * (copyOutParams.blockLen + copyOutParams.srcStride * 32)/ sizeof(DataType);
        ATVC::SyncDataQueue<AscendC::HardEvent::MTE3_MTE2>();

        CopyIn<DataType>(y, copyOutOffset, copyOutParams);
        AscendC::PipeBarrier<PIPE_V>();  // wait broadcast finished
        ATVC::SyncDataQueue<AscendC::HardEvent::MTE2_V>();

        Compute<DataType>(y, z, x, copyOutOffset, size);
        ATVC::SyncDataQueue<AscendC::HardEvent::V_MTE3>(); 
            
        CopyOut<DataType>(z, copyOutOffset, copyOutParams);            
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    template <class... Args>
    __aicore__ inline void InitArgs(GM_ADDR src, GM_ADDR dst)
    {
        srcGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(src));
        dstGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(dst));
    }

    template<typename DataType>
    __aicore__ inline void CopyIn(AscendC::LocalTensor<DataType> y, uint32_t copyOutOffset,  AscendC::DataCopyExtParams &copyOutParams)
    {
        AscendC::DataCopyPadExtParams<DataType> padParams{false, 0, 0, 0};
        uint32_t tmp = copyOutParams.srcStride;
        copyOutParams.srcStride = copyOutParams.dstStride;
        copyOutParams.dstStride = tmp;
        AscendC::DataCopyPad(y, srcGlobal_[copyOutOffset], copyOutParams, padParams);
        copyOutParams.dstStride = copyOutParams.srcStride;
        copyOutParams.srcStride = tmp;
    }

    template<typename DataType>
    __aicore__ inline void Compute(AscendC::LocalTensor<DataType> y, AscendC::LocalTensor<DataType> z, AscendC::LocalTensor<DataType> x,
        uint32_t copyOutOffset, uint32_t size)
    {
        AscendC::Add(z, x, y, size);     
    }

    template<typename DataType>
    __aicore__ inline void CopyOut(AscendC::LocalTensor<DataType> z, uint32_t copyOutOffset,  AscendC::DataCopyExtParams &copyOutParams)
    {
        AscendC::DataCopyPad(dstGlobal_[copyOutOffset], z, copyOutParams);
    }

    AscendC::GlobalTensor<DataType> srcGlobal_;
    AscendC::GlobalTensor<DataType> dstGlobal_;
};
#endif
