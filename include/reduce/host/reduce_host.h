/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_REDUCE_HOST_H
#define ATVC_REDUCE_HOST_H
#include "common/atvc_opdef.h"
#include "common/atvc_op_check.h"
#include "common/dtype_utils.h"
#include "common/const_def.h"
#include "reduce/common/reduce_common.h"
#include "reduce/host/tiling/reduce_tiling.h"
#include "reduce/host/tiling/tiling_common.h"

namespace ATVC {
namespace Host {
/*!
 * \brief Validate the legitimacy of reduce tiling hyper param
 * \param[in] hyperParam, reduce tiling hyper param
 * \return bool result, return true if the hyper param is valid, otherwise return false.
 */
bool CheckReduceHyperParam(const ATVC::Host::ReduceTilingHyperParam &hyperParam)
{
    constexpr uint32_t MAX_BASE_LINE = 54 * 1024U;
    constexpr uint32_t MIN_BASE_LINE = 48 * 1024U;
    constexpr float MAX_INNER_A = 256;
    constexpr float MIN_INNER_A = 128;
    constexpr float EPSILON = 1e-6f;
    constexpr float MIN_THRESH_HOLD = 0.8f;
    constexpr float MAX_THRESH_HOLD = 0.95f;
    if(hyperParam.basicBlock > MAX_BASE_LINE || hyperParam.basicBlock < MIN_BASE_LINE) {
        printf("[ERROR]: [ATVC][Reduce] Tiling hyperParam is invalid: "
            "basicBlock(%u) must be in [48 * 1024, 54 * 1024].\n", hyperParam.basicBlock);
        return false;
    }
    if(hyperParam.maxInnerA > MAX_INNER_A || hyperParam.maxInnerA < MIN_INNER_A) {
        printf("[ERROR]: [ATVC][Reduce] Tiling hyperParam is invalid: "
            "maxInnerA(%u) must be in [128, 256].\n", hyperParam.maxInnerA);
        return false;
    }
    if(hyperParam.balanceThreshHold > MAX_THRESH_HOLD + EPSILON || hyperParam.balanceThreshHold + EPSILON < MIN_THRESH_HOLD) {
        printf("[ERROR]: [ATVC][Reduce] Tiling hyperParam is invalid: balanceThreshHold(%f) must be in [0.8, 0.95].\n",
            hyperParam.balanceThreshHold);
        return false;
    }

    return true;
}

/*!
 * \brief Calculate the TilingData and policy parameters for Reduce.
 * \param[in] inputShape, shape of the tensor.
 * \param[in] reduceDim, the dim that requires a Reduce operation.
 * \param[out] policy, static policy of Reduce Template
 * \param[out] param, dynamic param of Reduce Template
 * \return bool Return true to indicate calculation success, false to indicate failure.
 */
template<class OpTraits, class PreComputeTraits = void, class PostComputeTraits = void>
bool CalcReduceTiling(std::vector<int64_t> inputShape,
                      std::vector<int64_t> reduceDim,
                      ReducePolicy* policy,
                      ReduceParam* param,
                      ReduceTilingHyperParam hyperParam = ReduceTilingHyperParam())
{
    if (policy == nullptr || param == nullptr) {
        printf("[ERROR]: [ATVC][Reduce] Invalid input: policy or param is null pointer!\n");
        return false;
    }
    if (!CheckReduceHyperParam(hyperParam)) {
        printf("[ERROR]: [ATVC][Reduce] Invalid input: hyper param is invalid!\n");
        return false;
    }

    using inputDTypeList = typename OpTraits::In::types;
    static constexpr size_t inTensorSumBytes =
        ATVC::TypeListReduce<inputDTypeList, SizeValue<0>, SumSizes>::Type::VALUE;
    if (inTensorSumBytes == 0) {
        printf("[ERROR]: [ATVC][Reduce] Tiling Error: OpTraits Input cannot be null!\n");
        return false;
    }
    using DataType = typename ATVC::TypeListGet<inputDTypeList, 0>::Type;
    auto inputDtype = GetOriInputType<DataType>();
    if (inputDtype == ge::DataType::DT_UNDEFINED) {
        printf("[ERROR]: [ATVC][Reduce] Reduce template does not support this data type!\n");
        return false;
    }
    OpTiling::ReduceTilingInputParam opInput = {reduceDim, inputShape, inputDtype, GetPromoteDataType(inputDtype)};
    OpTiling::ReduceOpTiling<OpTraits, PreComputeTraits, PostComputeTraits> tiling(opInput, policy, param, hyperParam);
    if (tiling.Run() != 0) {
        printf("[ERROR]: [ATVC][Reduce] Run tiling failed!\n");
        return false;
    }
    return true;
};
} // Host
} // ATVC
#endif // ATVC_REDUCE_HOST_H