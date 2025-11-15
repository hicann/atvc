/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_PRE_COMPUTE_MUL_OF_BROADCAST_H
#define ATVC_PRE_COMPUTE_MUL_OF_BROADCAST_H

#include "broadcast/kernel/broadcast_kernel.h"

template<typename Traits>
struct PreComputeMulOfBroadcast {
    using Inputs = typename Traits::In::types;
    using Outputs = typename Traits::Out::types;
    using DataType = typename ATVC::TypeListGet<Inputs, 0>::Type;
    static constexpr size_t INPUT_SIZE = ATVC::TypeListSize<Inputs>::VALUE;

    template <class... Args>
    __aicore__ inline void SetParam(DataType value) 
    { 
        value_ = value;
    }

    template<class... Args>
    __aicore__ inline void SetArgs(Args... args)
    {
        InitArgsInput<0>(args...);
    }

    template<typename DataType>
    __aicore__ inline void operator()(AscendC::LocalTensor<DataType> tensor1, AscendC::LocalTensor<DataType> tensor2, AscendC::LocalTensor<DataType> temp1, AscendC::LocalTensor<DataType> temp2, 
        uint32_t copyInOffset, AscendC::DataCopyExtParams &copyInParams)
    {
        size_t size = copyInParams.blockCount * (copyInParams.blockLen + copyInParams.srcStride * 32) / sizeof(DataType);
        ATVC::SyncDataQueue<AscendC::HardEvent::MTE3_MTE2>();

        CopyIn<DataType>(tensor1, tensor2, copyInOffset, copyInParams);

	    AscendC::PipeBarrier<PIPE_V>();  // wait broadcast finished
        ATVC::SyncDataQueue<AscendC::HardEvent::MTE2_V>();

        AscendC::Mul(temp1, tensor1, tensor2, size);
        AscendC::Muls(temp2, temp1, value_, size);
    }

private:
    template <int32_t start, class... Args>
    __aicore__ inline void InitArgsInput(GM_ADDR x, Args... args)
    {
        input_[start].SetGlobalBuffer((__gm__ DataType*)x);
        if constexpr (start + 1 < INPUT_SIZE) {
            InitArgsInput<start + 1>(args...);
        }
    }

    template<typename DataType>
    __aicore__ inline void CopyIn(AscendC::LocalTensor<DataType> tensor1, AscendC::LocalTensor<DataType> tensor2, uint32_t copyInOffset,  AscendC::DataCopyExtParams &copyInarams)
    {
        AscendC::DataCopyPadExtParams<DataType> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(tensor1, input_[0][copyInOffset], copyInarams, padParams);
        AscendC::DataCopyPad(tensor2, input_[1][copyInOffset], copyInarams, padParams);
    }

    AscendC::GlobalTensor<DataType> input_[INPUT_SIZE];
    DataType value_;
};

#endif
