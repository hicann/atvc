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
 * \file reduce_buf_pool.h
 * \brief class ReduceBufPool
 */

#ifndef ATVC_REDUCE_BUF_POOL_H
#define ATVC_REDUCE_BUF_POOL_H

#include "kernel_operator.h"
#include "common/platform.h"
#include "common/ops_utils_device.h"

namespace ATVC {
namespace KernelUtils {
struct PoolManagerUnit {
    int32_t idx = -1;
    int32_t eleSize = 0;
    int32_t bufNum = 0;
    int32_t offset = 0;
};

class ReduceBufPool {
    constexpr static int32_t MAX_INPUT_SIZE = 10;

public:
    __aicore__ inline ReduceBufPool() {};

    template <typename T, class PromoteDataType>
    __aicore__ inline void Init(
        int32_t inputNum, int32_t computeNum, int32_t basicBlockLen, int32_t preNum, int32_t postNum)
    {
        constexpr int32_t inputEleSize = sizeof(T);
        constexpr int32_t computeEleSize = sizeof(PromoteDataType);
        basicNum_ = basicBlockLen / sizeof(T);
        int32_t poolSize = basicNum_ * inputEleSize * preNum + basicNum_ * inputEleSize * postNum +
                           basicNum_ * inputEleSize * inputNum + basicNum_ * computeEleSize * computeNum;

        preUnit_.bufNum = inputNum;
        preUnit_.eleSize = inputEleSize;
        preUnit_.offset = 0;

        postUnit_.bufNum = inputNum;
        postUnit_.eleSize = inputEleSize;
        postUnit_.offset = basicNum_ * sizeof(T) * preNum;

        inputUnit_.bufNum = inputNum;
        inputUnit_.eleSize = inputEleSize;
        inputUnit_.offset = basicNum_ * sizeof(T) * postNum + postUnit_.offset;
        computeUnit_.bufNum = computeNum;
        computeUnit_.eleSize = computeEleSize;
        computeUnit_.offset = basicNum_ * sizeof(T) * inputNum + inputUnit_.offset;
        // Init buffer
        GetTPipePtr()->InitBuffer(qQue_, poolSize);
        AscendC::LocalTensor<T> inputUb = qQue_.GetWithOffset<T>(basicNum_ * inputNum, 0);
        AscendC::Duplicate<T>(inputUb, 0, basicNum_ * inputNum);
    }

    __aicore__ inline void ResetEvent()
    {
        inputUnit_.idx = -1;
        computeUnit_.idx = -1;
    }

    template <typename T>
    __aicore__ inline void ResetInputSize(int32_t inputNum) {
        inputUnit_.bufNum = inputNum;
        computeUnit_.offset = basicNum_ * sizeof(T) * inputNum;
    }

    __aicore__ inline void ResetComputeSize(int32_t computeNum)
    {
        computeUnit_.bufNum = computeNum;
    }

    template <bool IsInput, bool needDup=true, bool temp=false, typename T>
    __aicore__ inline const void AllocTensor(AscendC::LocalTensor<T>& tensor) {
        if constexpr (temp){
            if constexpr (IsInput) {
                int32_t idx = GetPreTensorId();
                tensor = qQue_.GetWithOffset<T>(basicNum_, preUnit_.offset + idx * basicNum_ * sizeof(T));
            } else {
                int32_t idx = GetPostTensorId();
                tensor = qQue_.GetWithOffset<T>(basicNum_, postUnit_.offset + idx * basicNum_ * sizeof(T));
            }
        } else if constexpr (IsInput) {
            int32_t idx = GetInputTensorId();
            tensor = qQue_.GetWithOffset<T>(basicNum_, inputUnit_.offset + idx * basicNum_ * sizeof(T));
            if constexpr (needDup) {
                AscendC::Duplicate<T>(tensor, 0, basicNum_);
                event_t allocEventId = static_cast<event_t>(GetTPipePtr()->FetchEventID<AscendC::HardEvent::V_MTE2>());
                eventIdV2Mte2_[idx] = allocEventId;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(allocEventId);
            }
        } else {
            int32_t idx = GetComputeTensorId();
            tensor = qQue_.GetWithOffset<T>(basicNum_, computeUnit_.offset + idx * basicNum_ * sizeof(T));
        }
    }

    template <typename T>
    __aicore__ inline void FreeTensor(const AscendC::LocalTensor<T>& tensor) {
        return;
    }

    template <typename T>
    __aicore__ inline void SyncTensor(const AscendC::LocalTensor<T>& tensor) {
        AscendC::LocalTensor<T> tmpBuf = qQue_.GetWithOffset<T>(basicNum_, 0);
        uint64_t start = (uint64_t)(tmpBuf.GetPhyAddr());
        uint64_t offset = (uint64_t)(tensor.GetPhyAddr());
        if (offset - start < computeUnit_.offset) {
            int32_t idx = (offset - start) / sizeof(T) / basicNum_;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdV2Mte2_[idx]);
        }
    }

private:
    __aicore__ inline int32_t GetComputeTensorId()
    {
        computeUnit_.idx = (computeUnit_.idx + 1) % computeUnit_.bufNum;
        return computeUnit_.idx;
    }

    __aicore__ inline int32_t GetInputTensorId()
    {
        inputUnit_.idx = (inputUnit_.idx + 1) % inputUnit_.bufNum;
        return inputUnit_.idx;
    }

    __aicore__ inline int32_t GetPreTensorId()
    {
        preUnit_.idx = (preUnit_.idx + 1) % preUnit_.bufNum;
        return preUnit_.idx;
    }
    __aicore__ inline int32_t GetPostTensorId()
    {
        postUnit_.idx = (postUnit_.idx + 1) % postUnit_.bufNum;
        return postUnit_.idx;
    }

    bool memo_[MAX_INPUT_SIZE] = {0};
    PoolManagerUnit preUnit_;
    PoolManagerUnit inputUnit_;
    PoolManagerUnit computeUnit_;
    PoolManagerUnit postUnit_;
    event_t eventIdV2Mte2_[MAX_INPUT_SIZE];
    AscendC::TBuf<> qQue_;
    int32_t basicNum_;
};  // class ReduceBufPool
} // namespace KernelUtils
} // namespace ATVC
#endif
