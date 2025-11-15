/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_reduce_sum_custom.h"

namespace {
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // Fixed code, acl initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return 1);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return 1);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return 1);

    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return 1);

    // Call aclrtMemcpy to copy host data to device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return 1);

    // Call aclCreateTensor to create a aclTensor object
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

void DestroyResources(std::vector<void *> tensors, std::vector<void *> deviceAddrs, aclrtStream stream,
                      int32_t deviceId, void *workspaceAddr = nullptr)
{
    // Release aclTensor and device
    for (uint32_t i = 0; i < tensors.size(); i++) {
        if (tensors[i] != nullptr) {
            aclDestroyTensor(reinterpret_cast<aclTensor *>(tensors[i]));
        }
        if (deviceAddrs[i] != nullptr) {
            aclrtFree(deviceAddrs[i]);
        }
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    // Destroy stream and reset device
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

void InitializeData(std::vector<float> &inputX, std::vector<float> &outputY, std::vector<float> &golden,
    std::vector<int64_t> &inputXShape, std::vector<int64_t> &outputYShape)
{
    for (int i = 0; i < inputXShape[0] * inputXShape[1]; ++i) {
        inputX[i] = 1.0;
    }
    float dealResult = 8.0;
    for (int i = 0; i < outputYShape[0] * outputYShape[1]; ++i) {
        golden[i] = dealResult;
        outputY[i] = 0.0;
    }
}

bool VerifyResults(const std::vector<float> &goldenData, const std::vector<float> &resultData)
{
    int64_t len = 10;
    LOG_PRINT("result is:\n");
    for (int64_t i = 0; i < len; i++) {
        LOG_PRINT("%.1f ", resultData[i]);
    }
    LOG_PRINT("\n");
    if (std::equal(resultData.begin(), resultData.end(), goldenData.begin())) {
        LOG_PRINT("test pass\n");
    } else {
        LOG_PRINT("test failed\n");
        return false;
    }
    return true;
}
}

int main(int argc, char **argv)
{
    // 1. (Fixed code) Initialize device / stream, refer to the list of external interfaces of acl
    // Update deviceId to your own device id
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return 1);

    // 2. Create input and output, need to customize according to the interface of the API
    std::vector<int64_t> inputXShape = {8, 2048}; 
    std::vector<int64_t> outputYShape = {1, 2048};
    void *inputXDeviceAddr = nullptr;  
    void *outputYDeviceAddr = nullptr;
    aclTensor *inputX = nullptr; 
    aclTensor *outputY = nullptr;
    std::vector<float> inputXHostData(inputXShape[0] * inputXShape[1]); 
    std::vector<float> outputYHostData(outputYShape[0] * outputYShape[1]); 
    std::vector<float> goldenData(outputYShape[0] * outputYShape[1]);
 
    InitializeData(inputXHostData, outputYHostData, goldenData, inputXShape, outputYShape);
    std::vector<void *> tensors = {inputX, outputY};
    std::vector<void *> deviceAddrs = {inputXDeviceAddr, outputYDeviceAddr};
    // Create inputX aclTensor
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return 1);
    // Create outputY aclTensor
    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return 1);
    // Create dimOut aclIntArray
    std::vector<int64_t> dim{0};
    aclIntArray* dimOut = aclCreateIntArray(dim.data(), dim.size());
    // 3. Call the API of the custom operator library
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Calculate the workspace size and allocate memory for it
    ret = aclnnReduceSumCustomGetWorkspaceSize(inputX, dimOut, outputY, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceSumCustomGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId); return 1);   
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return 1);
    }
    // Execute the custom operator
    ret = aclnnReduceSumCustom(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceSumCustom failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return 1);

    // 4. (Fixed code) Synchronize and wait for the task to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return 1);

    // 5. Get the output value, copy the result from device memory to host memory, need to modify according to the
    // interface of the API
    auto size = GetShapeSize(outputYShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputYDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return 1);

    // 6. Detroy resources, need to modify according to the interface of the API
    DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr);

    // print the output result
    if (!VerifyResults(goldenData, resultData)) {        
        return -1;
    }
    return 0;
}
