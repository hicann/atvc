/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_ELE_WISE_COMMON_H
#define ATVC_ELE_WISE_COMMON_H

namespace ATVC {
struct EleWiseTilingData {
    uint32_t tailBlockCnt;  // The number of cores that need to execute an additional loop
    uint32_t tailElemCnt;   // The number of tail block elements
    uint32_t numPerBlock;   // The number of basic blocks to be calculated for each core
    uint32_t tiledCnt;      // The number of basic block elements
    uint32_t blockNum;      // Execute audit
};

/*!
 * \brief EleWiseParam dynamic param of EleWise Template.
 */
struct EleWiseParam {
    EleWiseTilingData tilingData;  // Related parameters affecting data handling
    uint32_t totalCnt = 0;         // The number of elements in a single Tensor
    uint32_t nBufferNum = 2;       // The number of Tensors in each queue
};
}  // namespace ATVC

#endif
