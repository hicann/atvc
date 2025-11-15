/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_TILING_COMMON_H
#define ATVC_TILING_COMMON_H
#include "reduce/common/patterns.h"

namespace OpTiling {
constexpr static int32_t CACHE_SIZE = 16 * 1024;  // cahce size for ub reduce
constexpr static int32_t A_STEP_LEN = 4;

struct ReduceTilingUnit {
    int32_t idx = -1;    // ub cut axis
    uint64_t inner = 1;  // inner size in ub
    uint64_t outer = 1;  // outer size of ub
    uint64_t step = 1;   // step of cacheline
    void Update(int32_t idx, uint64_t inner, uint64_t outer, uint64_t step)
    {
        this->idx = idx;
        this->inner = inner;
        this->outer = outer;
        this->step = step;
    }
};

struct CacheLineBlock {
    int32_t axis = -1;            // cacheline cut axis
    uint64_t size = 1;            // cacheline size
    uint64_t cacheLineStep = 1;   // cacheline cut size for axis
    uint64_t cacheLineOuter = 1;  // relative to cacheLineStep, out size of cacheline cut axis
    uint64_t aSize = 1;           // A axis size in cacheline
};

struct ReduceTilingInputParam {
    std::vector<int64_t> reduceDim = {};
    std::vector<int64_t> reduceShape = {};
    ge::DataType inputDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType promoteDtpye = ge::DataType::DT_UNDEFINED;
    ReduceTilingInputParam(std::vector<int64_t> reduceDim_, std::vector<int64_t> reduceShape_, ge::DataType inputDtype_,
        ge::DataType promoteDtpye_)
        : reduceDim(reduceDim_), reduceShape(reduceShape_), inputDtype(inputDtype_), promoteDtpye(promoteDtpye_)
    {}
};

void MakeWrapDim(const std::vector<int64_t> &shape, std::vector<int64_t> &axes)
{
    // EnsureNotScalar at least return 1-D Tensor, so shapeSize cannot be 0
    size_t shapeSize = shape.size();
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] < 0) {
            axes[i] += shapeSize;
        }
    }
}

template <class Pattern>
bool IsAxisA(int32_t idx)
{
    if (Pattern::FirstA) {
        return idx % ATVC::CONST2 == 0;
    } else {
        return idx % ATVC::CONST2 == 1;
    }
}

int32_t IsAxesValid(const std::vector<int64_t> &shape, const std::vector<int64_t> &axes)
{
    size_t shapeSize = shape.size();
    size_t axesSize = axes.size();
    if (axesSize > shapeSize) {
        printf("[ERROR]: [ATVC][Reduce] Axis size is greater than shape size.\n");
        return -1;
    };

    for (size_t i = 0; i < axesSize; i++) {
        if (axes[i] >= static_cast<int64_t>(shapeSize) || axes[i] < 0) {
            printf("[ERROR]: [ATVC][Reduce] Axis size incorrect.\n");
            return -1;
        };
    }
    return 0;
}

template <class Pattern>
bool IsEmtpyTensor(const std::vector<uint64_t> &shape)
{
    for (int32_t i = 0; i < Pattern::Dim; i++) {
        if (shape[i] == 0) {
            return true;
        }
    }
    return false;
}

};  // namespace OpTiling

namespace ATVC {
namespace Host {
// Hyper param for reduce tiling.
struct ReduceTilingHyperParam {
    // Set the basic block memory size for Reduce, generally not exceeding 1/3 of the memory. It is recommended to set
    // it between [48k-54k]
    uint32_t basicBlock = 48 * 1024;
    uint32_t maxInnerA = 128;         // [128, 256]
    double balanceThreshHold = 0.85;  // Threshold level for multi-core equilibrium [0.8-0.95]
};
}  // namespace Host
}  // namespace ATVC
#endif  // ATVC_TILING_COMMON_H