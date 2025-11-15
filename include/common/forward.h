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
 * \file forward.h
 * \brief
 */
#ifndef ATVC_FORWARD_H
#define ATVC_FORWARD_H

#include <type_traits>

namespace ATVC {
template <typename T>
__aicore__ inline constexpr T&& Forward(std::remove_reference_t<T>& param) noexcept
{
    return static_cast<T&&>(param);
}

template <typename T>
__aicore__ inline constexpr T&& Forward(std::remove_reference_t<T>&& param) noexcept
{
    return static_cast<T&&>(param);
}

} // namespace ATVC
#endif
