/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_EXAMPLE_COMMON_H
#define ATVC_EXAMPLE_COMMON_H
#include "acl/acl.h"

namespace {
#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0)

static constexpr float REL_TOL = 1e-3f;
static constexpr float ABS_TOL = 1e-3f;

bool IsClose(float a, float b)
{
    const float eps = 1e-40f;
    float diff = std::abs(a - b);
    return (diff <= ABS_TOL) || (diff <= REL_TOL * std::max(std::abs(a), std::abs(b) + eps));
}

bool VerifyResults(const std::vector<float> &golden, const std::vector<float> &output)
{
    for (int32_t i = 0; i < golden.size(); i++) {
        if (!IsClose(golden[i], output[i])) {
            printf("[ERROR]: Accuracy verification failed! The expected value of element "
                   "in index [%d] is %f, but actual value is %f.\n",
                i,
                golden[i],
                output[i]);
            return false;
        }
    }
    return true;
}

void InitializeACL(aclrtContext &context, aclrtStream &stream, int32_t deviceId)
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));
}

void CleanACL(aclrtStream &stream, aclrtContext &context, int32_t deviceId)
{
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}
}  // namespace

#endif