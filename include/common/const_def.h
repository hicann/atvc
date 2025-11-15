
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_COMMON_CONST_DEF_H
#define ATVC_COMMON_CONST_DEF_H
#include <cstdint>
namespace ATVC {
constexpr int32_t UB_ALIGN_32 = 32;
constexpr int32_t UB_ALIGN_31 = 31;
constexpr int32_t UB_ALIGN_255 = 255;
constexpr int32_t UB_ALIGN_256 = 256;
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;
constexpr int32_t DIM3 = 3;
constexpr int32_t DIM4 = 4;
constexpr int32_t DIM5 = 5;
constexpr int32_t DIM6 = 6;
constexpr int32_t DIM7 = 7;
constexpr int32_t DIM8 = 8;
constexpr int32_t DIM9 = 9; // reserved

constexpr int32_t CONST0 = 0;
constexpr int32_t CONST1 = 1;
constexpr int32_t CONST2 = 2;
constexpr int32_t CONST3 = 3;
constexpr int32_t CONST4 = 4;
constexpr int32_t CONST5 = 5;
constexpr int32_t CONST6 = 6;
constexpr int32_t CONST7 = 7;
constexpr int32_t CONST8 = 8;
constexpr int32_t CONST9 = 9;
constexpr int32_t CONST10 = 10;
constexpr int64_t CONST63 = 63;

constexpr int32_t MAX_DIM = 9;

constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024; // fixed workspace size for ascendc
constexpr uint64_t BLOCK_SIZE_32K = 32 * 1024;
constexpr uint64_t BLOCK_SIZE_48K = 48 * 1024;
constexpr uint64_t BLOCK_SIZE_64K = 64 * 1024;
} // namespace ATVC
#endif