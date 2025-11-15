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
 * \file reduce_op_template.h
 * \brief
 */
#ifndef ATVC_REDUCE_OP_TEMPLATE_H
#define ATVC_REDUCE_OP_TEMPLATE_H

#include <type_traits>
#include <tuple>
#include "common/const_def.h"
#include "common/kernel_check_debug.h"
#include "kernel_operator.h"
#include "reduce/common/patterns.h"
#include "reduce/common/reduce_common.h"
#include "reduce/kernel/utils/reduce_block_aux_util.h"
#include "reduce/kernel/utils/reduce_block_aux.h"
#include "reduce/kernel/utils/reduce_util.h"
#include "reduce/kernel/utils/reduce_buf_pool.h"


namespace ATVC {
namespace Kernel {
/*!
 * ReduceOpTemplate: Generic Reduce operator template.
 * Reduce operators usually refer to operators that perform reduction operations on elements in tensors,
 * such as summation and averaging. They can specify several dimensions for reduction calculations,
 * or reduce all elements to a scalar.
 */
template <class ReduceCompute, const auto& SelectReducePolicy, 
            class PreCompute = void, class PostCompute = void>
class ReduceOpTemplate {
public:
    // for v-v fusion
    constexpr static bool HAS_PRE_COMPUTE = !AscendC::Std::is_same_v<PreCompute, void>;
    constexpr static bool HAS_POST_COMPUTE = !AscendC::Std::is_same_v<PostCompute, void>;
    using PreComputeTraits = AscendC::Std::conditional_t<HAS_PRE_COMPUTE, typename GetFunctionTraits<PreCompute>::ComputeTraits, VoidComputeTraits>;
    using PostComputeTraits = AscendC::Std::conditional_t<HAS_POST_COMPUTE, typename GetFunctionTraits<PostCompute>::ComputeTraits, VoidComputeTraits>;
    using PreInputs = typename PreComputeTraits::In::types;
    using PreOutputs = typename PreComputeTraits::Out::types;
    using PreTemp = typename PreComputeTraits::Temp::types;
    using PostInputs = typename PostComputeTraits::In::types;
    using PostOutputs = typename PostComputeTraits::Out::types;
    using PostTemp = typename PostComputeTraits::Temp::types;    
    constexpr static uint32_t ReduceInputCount = 1;
    constexpr static uint32_t ReduceOutputCount = 1;
    constexpr static size_t PreInputCount = ATVC::TypeListSize<PreInputs>::VALUE;
    constexpr static size_t PreOutputCount = ATVC::TypeListSize<PreOutputs>::VALUE;
    constexpr static size_t PreTempCount = ATVC::TypeListSize<PreTemp>::VALUE;
    constexpr static size_t PostInputCount = ATVC::TypeListSize<PostInputs>::VALUE;
    constexpr static size_t PostOutputCount = ATVC::TypeListSize<PostOutputs>::VALUE;
    constexpr static size_t PostTempCount = ATVC::TypeListSize<PostTemp>::VALUE;

    constexpr static ReduceSchLoopInfo SCH_LOOP_INFO = KernelUtils::Reduce::GetSchLoopInfo<SelectReducePolicy>();
    using Pattern = typename ReducePattern::GetPattern<SCH_LOOP_INFO.patternID>::T;
    using DataType = typename ReduceCompute::DataType;
    using PromoteDataType = typename ReduceCompute::PrompteDtype;
    constexpr static int32_t ELEMENT_ONE_REPEAT_ORI = static_cast<int32_t>(Platform::GetVRegSize()) / sizeof(DataType);
    constexpr static int32_t ELEMENT_ONE_REPEAT_COMPUTE =
        static_cast<int32_t>(Platform::GetVRegSize()) / sizeof(PromoteDataType);
    constexpr static int32_t VL_LENGTH_B = static_cast<int32_t>(Platform::GetVRegSize());
    constexpr static uint64_t BLOCK_SIZE_BYTE = Platform::GetUbBlockSize();
    constexpr static uint64_t CACHE_BUF_SIZE = 16 * 1024;
    constexpr static uint64_t RES_BUF_SIZE = 16 * 1024;
    constexpr static uint32_t T_BUF_SIZE = KernelUtils::GetCopyInCount<DataType>();
    constexpr static uint32_t PROMOTE_BUF_SIZE = KernelUtils::GetComputeCount<DataType>();
    AscendC::LocalTensor<PromoteDataType> tempBuf_;
    AscendC::LocalTensor<PromoteDataType> computeRes_;
    // The calculation object passed in by user
    ReduceCompute compute_;

public:
    __aicore__ inline ReduceOpTemplate() {}

    /*!
     * \brief The input order is: input tensor, output tensor, ReduceParam, Other scalars.
     *        Internally schedule data based on ReduceParam and call ReduceOpTemplate to complete
     *        the calculation before moving it out to GM.
     * \param[in] x, GM address of the input tensor.
     * \param[in] y, GM address of the output tensor.
     * \param[in] param, tiling data and policy.
     * \return void.
     */
    template<typename ...Args> 
    __aicore__ inline void Run(Args&&... args) 
    { 
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Start to run template function.\n");
        constexpr size_t PRE_ARGS_COUNT = HAS_PRE_COMPUTE ? PreInputCount + PreOutputCount - ReduceInputCount : 0;
        constexpr size_t REDUCE_ARGS_COUNT = ReduceInputCount + ReduceOutputCount - HAS_PRE_COMPUTE - HAS_POST_COMPUTE;
        constexpr size_t POST_ARGS_COUNT = HAS_POST_COMPUTE ? PostInputCount + PostOutputCount - ReduceOutputCount : 0;
        auto tuple = AscendC::Std::forward_as_tuple(args...);
        
        SplitAndCall<PRE_ARGS_COUNT, REDUCE_ARGS_COUNT, POST_ARGS_COUNT>(tuple,
            AscendC::Std::make_index_sequence<PRE_ARGS_COUNT>{},
            AscendC::Std::make_index_sequence<REDUCE_ARGS_COUNT>{},
            AscendC::Std::make_index_sequence<POST_ARGS_COUNT>{},
            AscendC::Std::make_index_sequence<sizeof...(args) - PRE_ARGS_COUNT - REDUCE_ARGS_COUNT - POST_ARGS_COUNT>{}
        );
        KernelUtils::PrintReduceParam<ATVC::ReduceParam, SelectReducePolicy>(param_);
        if (!KernelUtils::CheckReduceParam<ATVC::ReduceParam>(param_)) {
            return;
        }
        Init((GM_ADDR)(param_->workspaceAddr));
        Process();
        pipeIn.Destroy();
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce] Template function execution completed.\n");
    }

public:
    /*!
     * \brief Initialise all pipes, queues and buffers.
     * \param[in] workspace, GM address for the workspace buffer.
     */
    __aicore__ inline void Init(GM_ADDR workspace)
    {
        if constexpr (!HAS_PRE_COMPUTE) {
            input_[0].SetGlobalBuffer((__gm__ DataType *)src_);
        }
        if constexpr (!HAS_POST_COMPUTE) {
            output_[0].SetGlobalBuffer((__gm__ DataType *)dst_);
        }
        InitArgsWorkspace(workspace);
        GetTPipePtr()->InitBuffer(tempResQue_, RES_BUF_SIZE);
        computeRes_ = tempResQue_.Get<PromoteDataType>();
        GetTPipePtr()->InitBuffer(tempBufQue_, CACHE_BUF_SIZE);

        tempBuf_ = tempBufQue_.template Get<PromoteDataType>();

        GetTPipePtr()->InitBuffer(tempUbQue_, BLOCK_SIZE_BYTE);
    }

    /*!
     * \brief Allocate a tensor from the internal buffer pool.
     * \tparam IsInput, true  – tensor will be used as input (read-only)
     *                  false – tensor will be used as output (read-write)
     * \tparam needDup, if true the buffer is duplicated (for double-buffering).
     * \param[in] tensor, localTensor reference that receives the allocation.
     */
    template <bool IsInput, bool needDup = true, bool temp = false, typename T>
    __aicore__ inline void AllocTensorAux(AscendC::LocalTensor<T>& tensor)
    {
        bufPool_.AllocTensor<IsInput, needDup, temp>(tensor);
    }

    /*!
     * \brief Release a tensor back to the buffer pool.
     * \param[in] tensor, LocalTensor to free.
     */
    template <typename T>
    __aicore__ inline void FreeTensorAux(const AscendC::LocalTensor<T>& tensor)
    {
        bufPool_.FreeTensor(tensor);
    }

    /*!
     * \brief Execute the reduction schedule.
     */
    template <class... Args>
    __aicore__ inline void Process(Args... args)
    {
        if constexpr (SelectReducePolicy.patternID == ATVC::AR_PATTERN::A && !HAS_PRE_COMPUTE && !HAS_POST_COMPUTE) {
            CopyInput2Output();
            return;
        }
        if constexpr (SCH_LOOP_INFO.loopRCount == 0) {
            using SchTypeA = KernelUtils::Reduce::ReduceBlockAux<
                ATVC::ReduceTilingData, &SCH_LOOP_INFO,
                std::remove_reference_t<decltype(*this)>, DataType, DataType,
                PreCompute, PostCompute>;

            SchTypeA op(this, input_, output_, &(this->param_)->tilingData);
            op.Process(args...);
        } else {
            // Complete the first phase of Reduce
            using SchTypeR = KernelUtils::Reduce::ReduceBlockAux<
                ATVC::ReduceTilingData, &SCH_LOOP_INFO,
                std::remove_reference_t<decltype(*this)>, DataType,
                PromoteDataType, PreCompute, void>;
            SchTypeR op(this, input_, &workspace_, &(this->param_)->tilingData);
            op.Process(args...);
            bufPool_.ResetEvent();

            // Full nuclear synchronization
            AscendC::SyncAll();

            // Complete the second phase of Reduce
            bufPool_.template ResetInputSize<PromoteDataType>(3);
            constexpr static ReduceSchLoopInfo groupSchLoopInfo = KernelUtils::Reduce::GetGroupSchLoopInfo();
            ATVC::ReduceTilingData groupTiling;
            SetGroupTiling(groupTiling);
            using SchTypeA = KernelUtils::Reduce::ReduceBlockAux<ATVC::ReduceTilingData, &groupSchLoopInfo,
                                                                 std::remove_reference_t<decltype(*this)>,
                                                                 PromoteDataType, DataType, void, PostCompute>;
            SchTypeA groupOp(this, &workspace_, output_, &groupTiling);
            groupOp.Process(args...);
        }
    }

    /*!
     * \brief Populate tiling data for the second (group) reduction phase.
     * \param[in] groupTiling, tiling structure to be filled.
     * \return void.
     */
    __aicore__ inline void SetGroupTiling(ATVC::ReduceTilingData& groupTiling)
    {
        groupTiling.ubFactorA = ELEMENT_ONE_REPEAT_COMPUTE;
        groupTiling.ubFactorR = this->param_->tilingData.groupR;
        groupTiling.shape[0] = this->param_->tilingData.groupR;
        groupTiling.shape[1] = this->param_->tilingData.outSize;
        groupTiling.stride[0] = this->param_->tilingData.outSize;
        groupTiling.stride[1] = 1;
        groupTiling.dstStride[0] = this->param_->tilingData.outSize;
        groupTiling.dstStride[1] = 1;
        groupTiling.groupR = 1;
        groupTiling.outSize = this->param_->tilingData.outSize;
        groupTiling.factorRCntPerCore = 1;
        groupTiling.factorRTotalCnt = 1;
        groupTiling.factorATotalCnt = OpsUtils::CeilDiv(groupTiling.shape[1], groupTiling.ubFactorA);
        groupTiling.factorACntPerCore = OpsUtils::CeilDiv(groupTiling.factorATotalCnt,
                                                          static_cast<uint64_t>(64));  // 按照64核计算，需要tiling传
    }

    /*!
     * \brief Copy input tensor to UB with optional padding.
     * \tparam isPadding, true  – perform padding using PreCompute::GetPaddingValue
     *                    false – no padding
     * \param[in] src, globalTensor source in GM.
     * \param[in] view, view descriptor describing the copy geometry.
     * \param[in] shape, shape descriptor (modified when padding).
     * \param[in] ubTensor, localTensor destination in UB.
     */
    template <bool isPadding, bool BLOCK_PRE_COMPUTE, class EleT, class ViewDescT, class ShapeDescT>
    __aicore__ inline void CopyInAux(const AscendC::GlobalTensor<EleT> &src,
                                     ViewDescT &view, ShapeDescT &shape,
                                     const AscendC::LocalTensor<EleT> &ubTensor)
    {
        EleT paddingValue = compute_.template GetPaddingValue<EleT>();
        uint8_t padCnt = ((view.axis[0].dstStride - view.burstLen) * sizeof(EleT)) % BLOCK_SIZE_BYTE / sizeof(EleT);
        int32_t dupliCnt = view.axis[0].dstStride * view.axis[0].repeat;
        AscendC::DataCopyPadExtParams<EleT> padParams{true, 0, padCnt, paddingValue};
        if constexpr (!isPadding) {
            padParams = {false, 0, 0, 0};
            shape.oriBurstLen = view.burstLen;
        }
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = view.axis[0].repeat;
        copyInParams.blockLen = view.burstLen * sizeof(EleT);                              // unit Byte
        copyInParams.srcStride = (view.axis[0].srcStride - view.burstLen) * sizeof(EleT);  // unit Byte
        copyInParams.dstStride = (view.axis[0].dstStride - view.burstLen) * sizeof(EleT) / BLOCK_SIZE_BYTE;  // unit block(32byte)
            ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce][CopyIn] Padding flag is %d, padding value is %d, block length is %u,"
            " repeat count is %u.\n", isPadding, paddingValue, copyInParams.blockLen, copyInParams.blockCount);
        bufPool_.SyncTensor(ubTensor);

        const int32_t repeats[CONST6] = {static_cast<int32_t>(view.axis[CONST1].repeat),
            static_cast<int32_t>(view.axis[CONST2].repeat), static_cast<int32_t>(view.axis[CONST3].repeat),
            static_cast<int32_t>(view.axis[CONST4].repeat), static_cast<int32_t>(view.axis[CONST5].repeat),
            static_cast<int32_t>(view.axis[CONST6].repeat)};
        const int32_t dstStrides[CONST6] = {static_cast<int32_t>(view.axis[CONST1].dstStride),
            static_cast<int32_t>(view.axis[CONST2].dstStride), static_cast<int32_t>(view.axis[CONST3].dstStride),
            static_cast<int32_t>(view.axis[CONST4].dstStride), static_cast<int32_t>(view.axis[CONST5].dstStride),
            static_cast<int32_t>(view.axis[CONST6].dstStride)};
        const int32_t srcStrides[CONST6] = {static_cast<int32_t>(view.axis[CONST1].srcStride),
            static_cast<int32_t>(view.axis[CONST2].srcStride), static_cast<int32_t>(view.axis[CONST3].srcStride),
            static_cast<int32_t>(view.axis[CONST4].srcStride), static_cast<int32_t>(view.axis[CONST5].srcStride),
            static_cast<int32_t>(view.axis[CONST6].srcStride)};

        int32_t total = 1;
        for (int32_t i = 0; i < CONST6; ++i) {
            total *= repeats[i];
        }

        for (int32_t idx = 0; idx < total; ++idx) {
            int32_t tmp = idx;
            int32_t dstOffset = 0;
            int32_t srcOffset = 0;
            for (int32_t axis = 0; axis < CONST6; ++axis) {
                int32_t coord = tmp % repeats[axis];
                tmp /= repeats[axis];
                dstOffset += coord * dstStrides[axis];
                srcOffset += coord * srcStrides[axis];
            }
            if constexpr (BLOCK_PRE_COMPUTE) {
                ProcessPreCompute(ubTensor, dstOffset, view.addr + srcOffset, copyInParams, padParams);
            } else {
                AscendC::DataCopyPad(ubTensor[dstOffset], src[view.addr + srcOffset], copyInParams, padParams);
            }
        } 
    }

    /*!
     * \brief Copy input tensor directly to output when no reduction is required.
     * \param void.
     * \return void.
     */
    __aicore__ inline void CopyInput2Output()
    {
        uint32_t shapeSize = 1;
        for (uint8_t i = 0; i < MAX_DIM; i++) {
            if (this->param_->tilingData.shape[i] <= 1) {
                break;
            }
            shapeSize = shapeSize * this->param_->tilingData.shape[i];
        }
        shapeSize = (shapeSize * sizeof(DataType) + UB_ALIGN_31) / UB_ALIGN_32 * UB_ALIGN_32 / sizeof(DataType);
        AscendC::LocalTensor<DataType> tmpLoc;
        AllocTensorAux<true, false>(tmpLoc);
        uint32_t locSize = tmpLoc.GetSize();
        uint32_t loopCnt = shapeSize / locSize *locSize;
        uint32_t tailCnt = shapeSize - loopCnt;
        for (uint32_t i = 0; i < loopCnt; i += locSize) {
            AscendC::DataCopy(tmpLoc, input_[0][i], locSize);
            event_t copyId = static_cast<event_t>(GetTPipePtr()->FetchEventID<AscendC::HardEvent::MTE2_MTE3>());
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyId);
            AscendC::DataCopy(output_[0][i], tmpLoc, locSize);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyId);
        }
        if (tailCnt > 0) {
            AscendC::DataCopy(tmpLoc, input_[0][loopCnt], tailCnt);
            event_t copyId = static_cast<event_t>(GetTPipePtr()->FetchEventID<AscendC::HardEvent::MTE2_MTE3>());
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyId);
            AscendC::DataCopy(output_[0][loopCnt], tmpLoc, tailCnt);
        }
    }

    template <bool needMask, class Pattern, class V, class T, class... Args>
    __aicore__ inline void ComputeAux(V& view, const AscendC::LocalTensor<T>& ubTensor, Args... args)
    {
        compute_.template Compute<needMask, Pattern>(view, computeRes_, ubTensor, args...);
    }

    template <bool BLOCK_POST_COMPUTE, class T, class U>
    __aicore__ inline void CopyOutAux(const AscendC::GlobalTensor<T>& dst, U& view, int64_t tmpBufOffset)
    {
        AscendC::DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
        copyOutParams.blockCount = view.axis[0].repeat;
        copyOutParams.blockLen = view.burstLen * sizeof(T);
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][Reduce][CopyOut] Block length is %u, repeat count is %u.\n",
                                  view.burstLen, copyOutParams.blockCount);
        AscendC::LocalTensor<DataType> outputLocal = tempBuf_[tmpBufOffset].template ReinterpretCast<DataType>();
        if constexpr (AscendC::IsSameType<PromoteDataType, DataType>::value) {

            SetEvent<AscendC::HardEvent::V_MTE3>(AscendC::HardEvent::V_MTE3);
            if constexpr (BLOCK_POST_COMPUTE) {
                ProcessPostCompute(tempBuf_[tmpBufOffset], view.addr, copyOutParams);
            } else {
                AscendC::DataCopyPad(dst[view.addr], tempBuf_[tmpBufOffset], copyOutParams);
            }
        } else {
            AscendC::LocalTensor<DataType> outputLocal = tempBuf_[tmpBufOffset].template ReinterpretCast<DataType>();
            AscendC::Cast(outputLocal, tempBuf_[tmpBufOffset], AscendC::RoundMode::CAST_RINT,
                    view.axis[0].repeat *
                        OpsUtils::CeilAlign(view.burstLen, static_cast<uint64_t>(ELEMENT_ONE_REPEAT_COMPUTE)));
            SetEvent<AscendC::HardEvent::V_MTE3>(AscendC::HardEvent::V_MTE3);
            if constexpr (BLOCK_POST_COMPUTE) {
                ProcessPostCompute(outputLocal, view.addr, copyOutParams);
            } else {
                AscendC::DataCopyPad(dst[view.addr], outputLocal, copyOutParams);
            }
        }
    }

    template <class T, class U>
    __aicore__ inline void CopyOutAuxGroup(const AscendC::GlobalTensor<T>& dst, U& view, int64_t tmpBufOffset)
    {
        SetEvent<AscendC::HardEvent::V_MTE3>(AscendC::HardEvent::V_MTE3);
        AscendC::DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
        copyOutParams.blockCount = view.axis[0].repeat;
        copyOutParams.blockLen = view.burstLen * sizeof(T);
        copyOutParams.srcStride = (view.axis[0].srcStride - view.burstLen) * sizeof(T) / BLOCK_SIZE_BYTE;
        AscendC::DataCopyPad(dst[view.addr], tempBuf_[tmpBufOffset], copyOutParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
    } 

private:
    template<size_t PRE_ARGS_COUNT, size_t REDUCE_ARGS_COUNT, size_t POST_ARGS_COUNT, typename Tuple,
        size_t... I1, size_t... I2, size_t... I3, size_t... I4>
    __aicore__ inline void SplitAndCall(Tuple&& t, AscendC::Std::index_sequence<I1...>, AscendC::Std::index_sequence<I2...>,
        AscendC::Std::index_sequence<I3...>, AscendC::Std::index_sequence<I4...>)
    {
        SetScalar(AscendC::Std::get<I4 + PRE_ARGS_COUNT + REDUCE_ARGS_COUNT + POST_ARGS_COUNT>(AscendC::Std::forward<Tuple>(t))...);
        bufPool_.template Init<DataType, PromoteDataType>(T_BUF_SIZE, PROMOTE_BUF_SIZE, basicBlockLen_, preCount, postCount);

        if constexpr (HAS_PRE_COMPUTE) {
            InitPreArgs(AscendC::Std::get<I1>(AscendC::Std::forward<Tuple>(t))...);
        }
        InitReduceArgs(AscendC::Std::get<I2 + PRE_ARGS_COUNT>(AscendC::Std::forward<Tuple>(t))...);
        if constexpr (HAS_POST_COMPUTE) {
            InitPostArgs(AscendC::Std::get<I3 + PRE_ARGS_COUNT + REDUCE_ARGS_COUNT>(AscendC::Std::forward<Tuple>(t))...);
        }
    }
    template <class... Args>
    __aicore__ inline void InitArgsOutput(void) {}

    template <class... Args>
    __aicore__ inline void InitArgsInput(void) {}

    template <class... Args>
    __aicore__ inline void InitArgsWorkspace(GM_ADDR workspace, Args... args)
    {
        workspace_.SetGlobalBuffer((__gm__ PromoteDataType *)workspace);
    }

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
    __aicore__ inline void InitReduceArgs(Args... args)
    {
        if constexpr (!HAS_PRE_COMPUTE) {
            InitArgsInput(args...);
        } else {
            InitArgsOutput(args...);
        }
    }

    template<class... Args>
    __aicore__ inline void InitPreArgs(Args... args)
    {
        for (int i = 0; i < preCount; i++) {
            AscendC::LocalTensor<DataType> tensor;
            AllocTensorAux<true, false, true>(tensor);
            preIn_[i] = tensor;
        }
        preCompute_.SetArgs(args...);
    }

    template<class... Args>
    __aicore__ inline void InitPostArgs(Args... args)
    {
        for (int i = 0; i < postCount; i++) {
            AscendC::LocalTensor<DataType> tensor;
            AllocTensorAux<false, false, true>(tensor);
            postOut_[i] = tensor;
        }
        postCompute_.SetArgs(args...);
    }

    template <class T1, class... Args>
    __aicore__ inline void SetScalar(T1 param, Args... args)
    {
        param_ = param;
        basicBlockLen_ = this->param_->tilingData.basicBlock;
        static constexpr int size = sizeof...(args);
        if constexpr (size == 0) {
            return;
        }
        if constexpr (HAS_PRE_COMPUTE) {
            preCompute_.SetScalar(args...);
        }
        if constexpr (HAS_POST_COMPUTE) {
            postCompute_.SetScalar(args...);
        }
    }

    template<int32_t idx, int32_t num, bool isPreCompute, class... Args>
    __aicore__ inline void AllocLocalTensors(Args... args)
    {
        if constexpr (idx < num) {
            AscendC::LocalTensor<DataType> tensor;
            if (isPreCompute) {
                 tensor = preIn_[idx];
            } else {
                 tensor = postOut_[idx];
            }
            AllocLocalTensors<idx + 1, num, isPreCompute>(tensor, args...);
            FreeTensorAux(tensor);
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
        AllocLocalTensors<0, preCount, true>(args...);
    }

    template<class... Args>
    __aicore__ inline void ProcessPostCompute(Args... args)
    {
        // first arg of args is the first input of postCompute, so only need to alloc PostInputCount - 1 tensors
        AllocLocalTensors<0, postCount, false>(args...);
    }

    ATVC::ReduceParam* param_;   // The runtime parameters calculated by CalcReduceTiling API
    AscendC::TPipe pipeIn;;
    AscendC::TBuf<> oriVecQue_;
    AscendC::TBuf<> tempResQue_;
    AscendC::TBuf<> tempBufQue_;
    AscendC::TBuf<> tempUbQue_;

    constexpr static int32_t Dim = Pattern::Dim;
    constexpr static uint32_t InputSize = 1;
    constexpr static uint32_t OutputSize = 1;

    AscendC::GlobalTensor<DataType> output_[OutputSize];
    AscendC::GlobalTensor<DataType> input_[InputSize];
    AscendC::GlobalTensor<PromoteDataType> workspace_;

    AscendC::LocalTensor<PromoteDataType> tempUb_;
 
    int64_t basicBlockLen_;
    int64_t oriBasicBlockLen_;

    KernelUtils::ReduceBufPool bufPool_;

    GM_ADDR src_;
    GM_ADDR dst_;
    AscendC::Std::conditional_t<HAS_PRE_COMPUTE, PreCompute, void*> preCompute_;
    AscendC::Std::conditional_t<HAS_POST_COMPUTE, PostCompute, void*> postCompute_;
    constexpr static uint32_t preCount = HAS_PRE_COMPUTE ? PreInputCount - 1 + PreOutputCount + PreTempCount : 0;
    constexpr static uint32_t postCount = HAS_POST_COMPUTE ? PostInputCount - 1 + PostOutputCount + PostTempCount : 0;
    AscendC::LocalTensor<PromoteDataType> preIn_[preCount];
    AscendC::LocalTensor<PromoteDataType> postOut_[postCount];
};
} // namespace Kernel
} // namespace ATVC
#endif // ATVC_REDUCE_OP_TEMPLATE_H
