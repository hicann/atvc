/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_CUSTOM_IMPL_H
#define ADD_CUSTOM_IMPL_H
#include "elewise/host/elewise_host.h"
#include "elewise/kernel/elewise_kernel.h"

using AddOpTraitsFloat = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;
using AddOpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int, int>, ATVC::OpOutputs<int>>;

// 传入编译态参数ATVC::OpTraits
template<typename Traits>
struct AddComputeFunc {
    /*
    函数说明： z = x + y
    参数说明：
        x                   : 参与运算的输入
        y                   : 参与运算的输入
        z                   : 参与运算的输出
    */
    template<typename T> 
    // 重载operator，提供给算子模板类调用
    __aicore__ inline void operator()(AscendC::LocalTensor<T> x, AscendC::LocalTensor<T> y, AscendC::LocalTensor<T> z) {
        AscendC::Add(z, x, y, z.GetSize()); // 开发调用AscendC Api自行实现计算逻辑, 通过c.GetSize()获取单次计算的元素数量
    }
};

/*
 * 该函数为Add算子核函数入口
 * x        Device上的gm地址，指向Add算子第一个输入
 * y        Device上的gm地址，指向Add算子第二个输入
 * z        Device上的gm地址，指向Add算子第一个输出
 * param    ATVC::EleWiseParam数据
 */
template <class Traits>
__global__ __aicore__ void AddCustom(GM_ADDR x, GM_ADDR y, GM_ADDR z, ATVC::EleWiseParam param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 将AddComputeFunc仿函数作为模板参数传入，实例化EleWiseOpTemplate模板类
    auto op = ATVC::Kernel::EleWiseOpTemplate<AddComputeFunc<Traits>>();
    op.Run(x, y, z, &param);
}
#endif
