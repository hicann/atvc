/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_BROADCAST_OP_TEMPLATE_H
#define ATVC_BROADCAST_OP_TEMPLATE_H
#include <type_traits>
#include <tuple>
#include "kernel_operator.h"
#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "common/kernel_check_debug.h"
#include "common/ops_utils_device.h"
#include "broadcast/kernel/utils/broadcast_buf_pool.h"
#include "broadcast/kernel/utils/broadcast_util.h"
#include "broadcast/kernel/broadcast_compute.h"

namespace ATVC {
struct BroadcastDataView {
    uint32_t dimASize;
    uint32_t dimBSize;
    uint32_t inShape[ATVC::MAX_DIM];
    uint32_t outShape[ATVC::MAX_DIM];
    uint32_t copyInSize;        // Single core copying data volume
    uint32_t A11;               // A11 actually involved in the calculation
    uint32_t A12;               // A12 actually participated in the calculation
    uint32_t B1;                // B1 who actually participated in the calculation
    uint32_t dimAOffset;        // The offset of input and output data in dimension A
    uint32_t dimBOffset;        // The offset of input and output data in the B dimension
    uint32_t copyOutBaseOffset; // Copy the base address of inter nuclear data
};

namespace Kernel {
/*!
 * BroadcastCompute: Used to perform element operations between tensors when their shapes are inconsistent.
 * By copying data in a dimension of length 1, the shapes of two tensors are aligned to support element wise
 * addition operations.
*/
template <class BroadcastCompute, const auto& SelectBroadcastPolicy, class PreCompute = void, class PostCompute = void>
class BroadcastOpTemplate {
public:
    // for v-v fusion
    static constexpr bool HAS_PRE_COMPUTE = !AscendC::Std::is_same_v<PreCompute, void>;
    static constexpr bool HAS_POST_COMPUTE = !AscendC::Std::is_same_v<PostCompute, void>;
    using PreComputeTraits = AscendC::Std::conditional_t<HAS_PRE_COMPUTE, typename GetFunctionTraits<PreCompute>::ComputeTraits, VoidComputeTraits>;
    using PostComputeTraits = AscendC::Std::conditional_t<HAS_POST_COMPUTE, typename GetFunctionTraits<PostCompute>::ComputeTraits, VoidComputeTraits>;
    using PreInputs = typename PreComputeTraits::In::types;
    using PreOutputs = typename PreComputeTraits::Out::types;
    using PreTemp = typename PreComputeTraits::Temp::types;
    using PostInputs = typename PostComputeTraits::In::types;
    using PostOutputs = typename PostComputeTraits::Out::types;
    using PostTemp = typename PostComputeTraits::Temp::types;
    using DataType = typename BroadcastCompute::DataType;

    static constexpr size_t PreInputCount = ATVC::TypeListSize<PreInputs>::VALUE;
    static constexpr size_t PreOutputCount = ATVC::TypeListSize<PreOutputs>::VALUE;
    static constexpr size_t PreTempCount = ATVC::TypeListSize<PreTemp>::VALUE;
    static constexpr size_t PostInputCount = ATVC::TypeListSize<PostInputs>::VALUE;
    static constexpr size_t PostOutputCount = ATVC::TypeListSize<PostOutputs>::VALUE;
    static constexpr size_t PostTempCount = ATVC::TypeListSize<PostTemp>::VALUE;
    static constexpr size_t BroadcastInputCount = 1;
    static constexpr size_t BroadcastOutputCount = 1;
    static constexpr uint32_t DATA_SIZE = sizeof(DataType);
    static constexpr uint32_t UB_ALIGN_COUNT = ATVC::UB_ALIGN_32 / DATA_SIZE;

    __aicore__ inline BroadcastOpTemplate() {}

    /*!
     * \brief The external running interface of BroadcastOpTemplate mainly completes resource initialization, 
     *        data migration, calculation scheduling and data migration operations
     * \param src, GM pointer for input data
     * \param dst, GM pointer for output data
     * \param broadcastParam, dynamic parameters of broadcast, including tiling data, workspace, etc
     */
    template<typename ...Args>
    __aicore__ inline void Run(Args&&... args)
    {
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Start to run template function.\n");
        constexpr size_t PRE_ARGS_COUNT = HAS_PRE_COMPUTE ? PreInputCount + PreOutputCount - BroadcastInputCount : 0;
        constexpr size_t BROADCAST_ARGS_COUNT = BroadcastInputCount + BroadcastOutputCount - HAS_PRE_COMPUTE - HAS_POST_COMPUTE;
        constexpr size_t POST_ARGS_COUNT = HAS_POST_COMPUTE ? PostInputCount + PostOutputCount - BroadcastOutputCount : 0;
        auto tuple = AscendC::Std::forward_as_tuple(AscendC::Std::forward<Args>(args)...);
        SplitAndCall<PRE_ARGS_COUNT, BROADCAST_ARGS_COUNT, POST_ARGS_COUNT>(tuple,
            AscendC::Std::make_index_sequence<PRE_ARGS_COUNT>{},
            AscendC::Std::make_index_sequence<BROADCAST_ARGS_COUNT>{},
            AscendC::Std::make_index_sequence<POST_ARGS_COUNT>{},
            AscendC::Std::make_index_sequence<sizeof...(args) - PRE_ARGS_COUNT - BROADCAST_ARGS_COUNT - POST_ARGS_COUNT>{}
        );
        // Check param
        ATVC::KernelUtils::PrintBroadcastParam<ATVC::BroadcastParam, SelectBroadcastPolicy>(param_);
        if (!ATVC::KernelUtils::CheckBroadcastParam<ATVC::BroadcastParam>(param_)) {
            return;
        }
        this->Process();
        pipeIn.Destroy();
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast] Template function execution completed.\n");
    }

    /*!
     * \brief Late parameter injection helper. In some fusion scenarios the host side needs to pass additional
     *        runtime parameters (e.g. fused-activation coefficients) after the kernel has already been launched.
     *        SetParam allows such parameters to be forwarded to the pre- and/or post-compute functors.
     * \param[in] param, pointer to the BroadcastParam structure (tiling, workspace, etc.)
     * \param[in] args, optional extra parameters consumed by PreCompute / PostCompute
     */
    template <class... Args>
    __aicore__ inline void SetParam(BroadcastParam *param, Args... args) 
    {
        param_ = param;
        tilingData_ = &param_->tilingData;
        static constexpr int size = sizeof...(args);
        if constexpr (size == 0) {
            return;
        }          
        if constexpr (HAS_PRE_COMPUTE) {
            preCompute_.SetParam(args...);
        }
        if constexpr (HAS_POST_COMPUTE) {
            postCompute_.SetParam(args...);
        }
    }

private:
    template<size_t PRE_ARGS_COUNT, size_t BROADCAST_ARGS_COUNT, size_t POST_ARGS_COUNT, typename Tuple,
        size_t... I1, size_t... I2, size_t... I3, size_t... I4>
    __aicore__ inline void SplitAndCall(Tuple&& t, AscendC::Std::index_sequence<I1...>, AscendC::Std::index_sequence<I2...>,
        AscendC::Std::index_sequence<I3...>, AscendC::Std::index_sequence<I4...>)
    {
        if constexpr (HAS_PRE_COMPUTE) {
            InitPreArgs(AscendC::Std::get<I1>(AscendC::Std::forward<Tuple>(t))...);
        }
        InitBroadcastArgs(AscendC::Std::get<I2 + PRE_ARGS_COUNT>(AscendC::Std::forward<Tuple>(t))...);
        if constexpr (HAS_POST_COMPUTE) {
            InitPostArgs(AscendC::Std::get<I3 + PRE_ARGS_COUNT + BROADCAST_ARGS_COUNT>(AscendC::Std::forward<Tuple>(t))...);
        }
        InitArgsParams(AscendC::Std::get<I4 + PRE_ARGS_COUNT + BROADCAST_ARGS_COUNT + POST_ARGS_COUNT>(AscendC::Std::forward<Tuple>(t))...);
    }

    template <class... Args>
    __aicore__ inline void InitArgsParams(Args... args)
    {
        SetParam(args...);
        if constexpr (!HAS_PRE_COMPUTE) {
            uint32_t srcDataSize = tilingData_->basicBlock;
            srcGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(src_), srcDataSize);
        }
        if constexpr (!HAS_POST_COMPUTE) {
            uint32_t dstDataSize = tilingData_->basicBlock;
            dstGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(dst_), dstDataSize);
        }
        inputCount_ = BroadcastInputCount;
        outputCount_ = BroadcastOutputCount;
        if (HAS_PRE_COMPUTE) {
            inputCount_ = PreInputCount + PreTempCount + PreOutputCount;  
        }
        if (HAS_POST_COMPUTE) {
            outputCount_ = PostInputCount + PostTempCount + PostOutputCount;
        }
        bufPool_.template Init<DataType>(inputCount_,  // The number of inputs required for double buffer
            outputCount_,  // The number of calculation results is generally consistent with inputNum
            tilingData_->A2 * tilingData_->A12 * DATA_SIZE,  // Input Tensor size
            tilingData_->A2 * tilingData_->B2 * DATA_SIZE);  // Output Tensor Size
    }

    template <class... Args>
    __aicore__ inline void InitArgsOutput(void) {}

    template <class... Args>
    __aicore__ inline void InitArgsInput(void) {}

    template <class... Args>
    __aicore__ inline void InitArgsOutput(GM_ADDR dst, Args... args)
    {
        dst_ = dst;
    }

    template <class... Args>
    __aicore__ inline void InitArgsInput(GM_ADDR src, Args... args)
    {
        src_ = src;
        InitArgsOutput(args...);
    }

    template<class... Args>
    __aicore__ inline void InitBroadcastArgs(Args... args)
    {
        InitArgsInput(args...);
    }

    template<class... Args>
    __aicore__ inline void InitPreArgs(Args... args)
    {
        preCompute_.SetArgs(args...);
    }

    template<class... Args>
    __aicore__ inline void InitPostArgs(Args... args)
    {
        postCompute_.SetArgs(args...);
    }

    template<int32_t idx, int32_t num, bool isPreCompute, class... Args>
    __aicore__ inline void AllocLocalTensors(Args... args)
    {
        if constexpr (idx < num) {
            AscendC::LocalTensor<DataType> tensor;
            bufPool_.template AllocTensor<false>(tensor);
            AllocLocalTensors<idx + 1, num, isPreCompute>(tensor, args...);
            bufPool_.template FreeTensor<false>(tensor);
        } else {
            if constexpr (isPreCompute) {
                preCompute_(args...);
            } else {
                postCompute_(args...);
            }
        }
    }

    template<class... Args>
    __aicore__ inline void ProcessPreCompute(Args... args)
    {
        // first arg of args is the first input of postCompute, so only need to alloc PostInputCount - 1 tensors
        constexpr int32_t localTensorCount = PreInputCount - 1 + PreOutputCount + PreTempCount;
        AllocLocalTensors<0, localTensorCount, true>(args...);
    }

    template<class... Args>
    __aicore__ inline void ProcessPostCompute(Args... args)
    {
        // first arg of args is the first input of postCompute, so only need to alloc PostInputCount - 1 tensors
        constexpr int32_t localTensorCount = PostInputCount - 1 + PostOutputCount + PostTempCount;
        AllocLocalTensors<0, localTensorCount, false>(args...);
    }

    __aicore__ inline void CopyOutBatch(BroadcastDataView &view,
                                        uint32_t dimACount,
                                        AscendC::LocalTensor<DataType> &output)
    {
        uint32_t dimBCount = 0;  
        for (int i = 0; i < view.B1; i++) {
            uint32_t copyOutOffset;
            if (SelectBroadcastPolicy.patternID == AB_PATTERN::ABA) {
                copyOutOffset = dimBCount * view.dimASize + dimACount * tilingData_->A2;
            } else {
                copyOutOffset = dimACount * tilingData_->A2 * view.dimBSize + dimBCount;
            }
            CopyOut(output, copyOutOffset + view.copyOutBaseOffset, view);
            dimBCount += tilingData_->B2;
        }
    }

    __aicore__ inline void Process()
    {
        BroadcastDataView view;
        CalcView(view);
        uint32_t inputOffset;
        uint32_t dimACount = 0;
        AscendC::LocalTensor<DataType> input;
        for (int i = 0; i < view.A11; i++) {
            inputOffset = 0;
            bufPool_.template AllocTensor<true>(input);
            uint32_t copyInOffset = i * view.A12 * tilingData_->A2;
            if (tilingData_->A0 != 1) {
                copyInOffset += view.dimAOffset;
            }
            if (copyInOffset >= view.dimASize) {
                return;
            }
            if (copyInOffset + view.copyInSize > view.dimASize) {
                // If the remaining data is not sufficient for a complete calculation,
                // recalculate based on the actual data
                view.copyInSize = view.dimASize - copyInOffset;
                view.A12 = OpsUtils::CeilDiv<uint32_t>(view.copyInSize, tilingData_->A2);
            }

            CopyIn(input, copyInOffset, view);
            bufPool_.template SetVecSync<DataType, AscendC::HardEvent::MTE2_V>(input);
            bufPool_.template WaitVecSync<DataType, AscendC::HardEvent::MTE2_V>(input);
            for (int j = 0; j < view.A12; j ++) {
                AscendC::LocalTensor<DataType> output;
                bufPool_.template AllocTensor<false>(output);

                compute_.template Compute<SelectBroadcastPolicy.patternID>(input, inputOffset, output,
                    OpsUtils::CeilAlign<uint32_t>(tilingData_->A2, UB_ALIGN_COUNT),
                    OpsUtils::CeilAlign<uint32_t>(tilingData_->B2, UB_ALIGN_COUNT));
                CopyOutBatch(view, dimACount, output);
                bufPool_.template FreeTensor<false>(output);
                dimACount++;
                inputOffset += tilingData_->A2;
                bufPool_.template SetCopyOutSync<DataType, AscendC::HardEvent::MTE3_V>(output);
                bufPool_.template WaitCopyOutSync<DataType, AscendC::HardEvent::MTE3_V>(output);
            }

            bufPool_.template FreeTensor<true>(input);            
        }
        bufPool_.ResetEvent();
    }

    __aicore__ inline uint32_t CalcCopyOutBaseOffset(BroadcastDataView &view)
    {
        uint32_t copyOutBaseOffset = 0;
        // Calculate the base offset for copying out
        if (SelectBroadcastPolicy.patternID == AB_PATTERN::ABA) {
            if (tilingData_->A0 != 1) { // If A is split across cores, add the partial-A offset.
                copyOutBaseOffset += view.dimAOffset;
            }
            if (tilingData_->B0 != 1) { // If B is split across cores, add the partial-B offset.
                copyOutBaseOffset += view.dimBOffset * view.dimASize;
            }
        } else {
            if (tilingData_->A0 != 1) { // If A is split across cores, add the partial-A offset.
                copyOutBaseOffset += view.dimAOffset * view.dimBSize;
            }
            if (tilingData_->B0 != 1) { // If B is split across cores, add the partial-B offset.
                copyOutBaseOffset += view.dimBOffset;
            }
        }    
        return copyOutBaseOffset;    
    }

    __aicore__ inline void CalcView(BroadcastDataView &view)
    {
        if (SelectBroadcastPolicy.patternID == AB_PATTERN::ABA) {
            view.dimASize = tilingData_->dstShape[1];
            view.dimBSize = tilingData_->dstShape[0];
            view.inShape[0] = 1;
            view.inShape[1] = tilingData_->A2;
            view.outShape[0] = tilingData_->B2;
            view.outShape[1] = tilingData_->A2;
        } else {
            view.dimASize = tilingData_->dstShape[0];
            view.dimBSize = tilingData_->dstShape[1];
            view.inShape[0] = tilingData_->A2;
            view.inShape[1] = 1;
            view.outShape[0] = tilingData_->A2;
            view.outShape[1] = tilingData_->B2;
        }        
        view.A11 = tilingData_->A11;
        view.A12 = tilingData_->A12;
        view.B1 = tilingData_->B1;
        uint32_t blockId = AscendC::GetBlockIdx();
        uint32_t dimAIdx = blockId / tilingData_->B0;
        uint32_t dimBIdx = blockId % tilingData_->factorBTotalCnt;
        view.dimAOffset = dimAIdx * tilingData_->factorACntPerCore;
        view.dimBOffset = dimBIdx * tilingData_->factorBCntPerCore;
        view.copyInSize = view.A12 * tilingData_->A2;
        if (view.dimAOffset + tilingData_->factorACntPerCore > view.dimASize) {
            // The remaining data in the A dimension is not enough to allocate the number of A values to each kernel.
            // Recalculate the actual A dimension segmentation.
            uint32_t realShape = view.dimASize - view.dimAOffset;
            uint32_t dimA1 =  OpsUtils::CeilDiv<uint32_t>(realShape, tilingData_->A2);
            if (dimA1 < view.A12) {
                view.A11 = 1;
                view.A12 = dimA1;
            } else {
                view.A11 = OpsUtils::CeilDiv<uint32_t>(dimA1, view.A12);
            }
        }
        if (view.dimBOffset + tilingData_->factorBCntPerCore > view.dimBSize) {
            uint32_t realShape = view.dimBSize - view.dimBOffset;
            view.B1 =  OpsUtils::CeilDiv<uint32_t>(realShape, tilingData_->B2);
        }
        view.copyOutBaseOffset = CalcCopyOutBaseOffset(view);
    }

    __aicore__ inline void CopyIn(AscendC::LocalTensor<DataType> &input, uint32_t copyInOffset, BroadcastDataView &view)
    {
        AscendC::DataCopyPadExtParams<DataType> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = view.copyInSize * DATA_SIZE;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        if constexpr(HAS_PRE_COMPUTE) {
            ProcessPreCompute(input, copyInOffset, copyInParams);
            return;
        }
        AscendC::DataCopyPad(input, srcGlobal_[copyInOffset], copyInParams, padParams);
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast][CopyIn] Offset is %u, block len is %u "
            "block count is %u.\n", copyInOffset, copyInParams.blockLen, copyInParams.blockCount);
    }

    __aicore__ inline void CopyOutNonAligned(AscendC::LocalTensor<DataType> &output,
                                             uint32_t copyOutOffset, BroadcastDataView &view)
    {
        uint32_t blockId = AscendC::GetBlockIdx();
        uint32_t dstDataSize = view.outShape[0] * view.outShape[1];
        uint64_t dstShape = tilingData_->dstShape[1];
        AscendC::DataCopyExtParams copyOutParams;
        copyOutParams.blockLen = view.outShape[1] * DATA_SIZE;
        copyOutParams.blockCount = dstDataSize * DATA_SIZE / copyOutParams.blockLen;
        copyOutParams.srcStride = 0;

        if (view.outShape[1] + copyOutOffset % dstShape > dstShape) {
            // Column is not aligned, copy according to actual data
            copyOutParams.srcStride = OpsUtils::CeilAlign<uint32_t>(view.outShape[1], UB_ALIGN_COUNT) * DATA_SIZE;
            copyOutParams.blockLen = (dstShape - copyOutOffset % dstShape) * DATA_SIZE;
            copyOutParams.srcStride = (copyOutParams.srcStride - copyOutParams.blockLen) / ATVC::UB_ALIGN_32;
        }
        if (view.outShape[0] + copyOutOffset / dstShape > tilingData_->dstShape[0]) {
            // Rows are not aligned, copy according to actual data
            copyOutParams.blockCount = (tilingData_->dstShape[0] - copyOutOffset / dstShape);
        }
        copyOutParams.dstStride = dstShape * DATA_SIZE - copyOutParams.blockLen;
        bufPool_.template SetCopyOutSync<DataType, AscendC::HardEvent::V_MTE3>(output);
        bufPool_.template WaitCopyOutSync<DataType, AscendC::HardEvent::V_MTE3>(output);
        if constexpr(HAS_POST_COMPUTE) {
            ProcessPostCompute(output, copyOutOffset, copyOutParams);
            return;
        }
        AscendC::DataCopyPad(dstGlobal_[copyOutOffset], output, copyOutParams);
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Broadcast][CopyOut] Offset is %u, block len is %u block count is %u.\n",
                                  copyOutOffset, copyOutParams.blockLen, copyOutParams.blockCount);
    }

    __aicore__ inline void CopyOut(AscendC::LocalTensor<DataType> &output, uint32_t copyOutOffset, BroadcastDataView &view)
    {
        CopyOutNonAligned(output, copyOutOffset, view);
    }

    GM_ADDR src_;
    GM_ADDR dst_;
    AscendC::TPipe pipeIn;
    AscendC::GlobalTensor<DataType> srcGlobal_;
    AscendC::GlobalTensor<DataType> dstGlobal_;
    BroadcastCompute compute_;
    AscendC::Std::conditional_t<HAS_PRE_COMPUTE, PreCompute, void*> preCompute_;
    AscendC::Std::conditional_t<HAS_POST_COMPUTE, PostCompute, void*> postCompute_;
    const BroadcastParam *param_;
    const BroadcastOpTilingData *tilingData_;
    KernelUtils::BroadcastBufPool<!HAS_PRE_COMPUTE && !HAS_POST_COMPUTE> bufPool_;
    uint32_t inputCount_;
    uint32_t outputCount_;
};
} // namespace Kernel
} // namespace ATVC
#endif // ATVC_BROADCAST_OP_TEMPLATE_H
