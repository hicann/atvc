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
 * \file broadcast_buf_pool.h
 * \brief class BroadcastBufPool
 */

#ifndef ATVC_BROADCAST_BUF_POOL_H
#define ATVC_BROADCAST_BUF_POOL_H

#include "kernel_operator.h"

namespace ATVC {
namespace KernelUtils {
struct BrcPoolManagerUnit {
    int32_t idx = -1;
    int32_t eleSize = 0;
    int32_t bufNum = 0;
    int32_t offset = 0;
};

template <bool EnableDb>
class BroadcastBufPool {
    constexpr static int32_t MAX_INPUT_SIZE = 10;

public:
    __aicore__ inline BroadcastBufPool() {};

    template <typename T>
    __aicore__ inline void Init(int32_t inputNum,  // The number of inputs required for doublebuff
        int32_t computeNum,  // The number of calculation results is generally consistent with inputNum
        int32_t inBlockLen,  // Basic input block size for one calculation
        int32_t outBlockLen)
    {
        // Basic input block size for one calculation
        /*
         _______________________________________________________________________________________________________
        |   inputTensor 0   |   inputTensor 1   |        outputTensor 0         |        outputTensor 0         |
        |___________________|___________________|_______________________________|_______________________________|
        */
        if (EnableDb) {
            inputNum *= ATVC::CONST2;
            computeNum *= ATVC::CONST2;
        }
        constexpr int32_t eleSize = static_cast<int32_t>(sizeof(T));
        inputNum_ = inBlockLen / eleSize;
        outputNum_ = outBlockLen / eleSize;
        int32_t poolSize = inBlockLen * inputNum + outBlockLen * computeNum;
        inputUnit_.bufNum = inputNum;
        inputUnit_.eleSize = eleSize;
        inputUnit_.offset = 0;
        computeUnit_.bufNum = computeNum;
        computeUnit_.eleSize = eleSize;
        computeUnit_.offset = inBlockLen * inputNum;
        // Init buffer
        GetTPipePtr()->InitBuffer(qQue_, poolSize);
    }

    template <bool IsInput, typename T>
    __aicore__ inline const void AllocTensor(AscendC::LocalTensor<T> &tensor)
    {
        if constexpr (IsInput) {
            int32_t idx = GetInputTensorId();
            tensor = qQue_.GetWithOffset<T>(inputNum_, inputUnit_.offset + idx * inputNum_ * sizeof(T));
        } else {
            int32_t idx = GetComputeTensorId();
            tensor = qQue_.GetWithOffset<T>(outputNum_, computeUnit_.offset + idx * outputNum_ * sizeof(T));
        }
    }

    template <bool IsInput, typename T>
    __aicore__ inline const void FreeTensor(AscendC::LocalTensor<T> &tensor)
    {
        if constexpr (!IsInput) {
            uint32_t idx = GetOutputTensorIdx<T>(tensor);
            isBusyOut_[idx] = false;  // Restore isBusy_ state
        }
    }

    template <typename T, AscendC::HardEvent EVENT>
    __aicore__ inline const void SetVecSync(AscendC::LocalTensor<T> &tensor)
    {
        uint32_t idx = GetInputTensorIdx<T>(tensor);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->AllocEventID<EVENT>());
        vecEventId_[idx] = eventId;
        AscendC::SetFlag<EVENT>(eventId);
    }

    template <typename T, AscendC::HardEvent EVENT>
    __aicore__ inline const void WaitVecSync(AscendC::LocalTensor<T> &tensor)
    {
        uint32_t idx = GetInputTensorIdx<T>(tensor);
        AscendC::WaitFlag<EVENT>(vecEventId_[idx]);
        GetTPipePtr()->ReleaseEventID<EVENT>(vecEventId_[idx]);
    }

    template <typename T, AscendC::HardEvent EVENT>
    __aicore__ inline const void SetCopyOutSync(AscendC::LocalTensor<T> &tensor)
    {
        uint32_t idx = GetOutputTensorIdx<T>(tensor);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->AllocEventID<EVENT>());
        outEventId_[idx] = eventId;
        AscendC::SetFlag<EVENT>(eventId);
    }

    template <typename T, AscendC::HardEvent EVENT>
    __aicore__ inline const void WaitCopyOutSync(AscendC::LocalTensor<T> &tensor)
    {
        uint32_t idx = GetOutputTensorIdx<T>(tensor);
        AscendC::WaitFlag<EVENT>(outEventId_[idx]);
        GetTPipePtr()->ReleaseEventID<EVENT>(outEventId_[idx]);
    }

    template <typename T>
    __aicore__ inline uint32_t GetInputTensorIdx(AscendC::LocalTensor<T> &tensor)
    {
        uint64_t start = (uint64_t)qQue_.GetWithOffset<T>(inputNum_, 0).GetPhyAddr();
        uint64_t offset = (uint64_t)tensor.GetPhyAddr();
        uint32_t idx = (offset - start) / sizeof(T) / inputNum_;
        return idx;
    }

    template <typename T>
    __aicore__ inline uint32_t GetOutputTensorIdx(AscendC::LocalTensor<T> &tensor)
    {
        uint64_t start = (uint64_t)qQue_.GetWithOffset<T>(outputNum_, inputNum_).GetPhyAddr();
        uint64_t offset = (uint64_t)tensor.GetPhyAddr();
        uint32_t idx = (offset - start) / sizeof(T) / outputNum_;
        return idx;
    }

    __aicore__ inline const void ResetEvent()
    {
        GetTPipePtr()->Reset();
    }

private:
    __aicore__ inline int32_t GetComputeTensorId()
    {
        uint32_t loopCnt = 0;
        do {
            computeUnit_.idx = (computeUnit_.idx + 1) % computeUnit_.bufNum;
            if (!isBusyOut_[computeUnit_.idx]) {
                break;
            }
            ++loopCnt;
        } while (
            // 10: At most 10 times can be searched, in fact, there is pipeline synchronization between
            // each cycle calculation and copying. Here, basically one cycle is enough
            loopCnt < ATVC::CONST10);

        isBusyOut_[computeUnit_.idx] = true;
        return computeUnit_.idx;
    }

    __aicore__ inline int32_t GetInputTensorId()
    {
        inputUnit_.idx = (inputUnit_.idx + 1) % inputUnit_.bufNum;
        return inputUnit_.idx;
    }

    BrcPoolManagerUnit inputUnit_;
    BrcPoolManagerUnit computeUnit_;
    event_t vecEventId_[MAX_INPUT_SIZE];
    event_t outEventId_[MAX_INPUT_SIZE];
    bool isBusyOut_[MAX_INPUT_SIZE] = {false};
    AscendC::TBuf<> qQue_;
    int32_t inputNum_;
    int32_t outputNum_;
};
}  // namespace KernelUtils
}  // namespace ATVC
#endif  // ATVC_BROADCAST_BUF_POOL_H
