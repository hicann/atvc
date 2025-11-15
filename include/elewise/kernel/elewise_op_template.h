/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_ELE_WISE_OP_TEMPLATE_H
#define ATVC_ELE_WISE_OP_TEMPLATE_H

#include <type_traits>
#include "kernel_operator.h"
#include "common/kernel_check_debug.h"
#include "common/tensor_info.h"
#include "common/type_list.h"
#include "common/index_seq.h"
#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "elewise/common/elewise_common.h"
#include "elewise/kernel/utils/elewise_util.h"

namespace ATVC {
namespace Kernel {
/*!
 * EleWiseOpTemplate: EleWiseOpTemplate provides templates for element level operations on tensors,
 * including but not limited to addition, subtraction, multiplication, division, as well as
 * mathematical functions such as exponentiation, logarithm, trigonometric functions, etc.
 * The characteristic of this type of operator is that it performs calculation operations
 * element by element without changing the shape of the input data.
 */
template <class EleWiseCompute>
class EleWiseOpTemplate {
    using EleWiseOpTraits = typename GetFunctionTraits<EleWiseCompute>::ComputeTraits;
    using Inputs = typename EleWiseOpTraits::In::types;
    using Outputs = typename EleWiseOpTraits::Out::types;
    using Temps = typename EleWiseOpTraits::Temp::types;
    // xCount represents how many tensors are in the tensorList
    static constexpr size_t INPUT_COUNT = ATVC::TypeListSize<Inputs>::VALUE;
    static constexpr size_t OUTPUT_COUNT = ATVC::TypeListSize<Outputs>::VALUE;
    static constexpr size_t TEMP_COUNT = ATVC::TypeListSize<Temps>::VALUE;
    // xxTensroSumbytes represents the cumulative length of all data types in tensorList,
    // xxTensroSumBytes = sum(sizeof(Tensor_i::type))
    static constexpr size_t IN_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Inputs, SizeValue<0>, SumSizes>::Type::VALUE;
    static constexpr size_t OUT_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Outputs, SizeValue<0>, SumSizes>::Type::VALUE;
    static constexpr size_t TEMP_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Temps, SizeValue<0>, SumSizes>::Type::VALUE;

public:
    __aicore__ inline EleWiseOpTemplate() {}
    /*!
     * \brief The external running interface of EleWiseOpTemplate mainly completes resource initialization, 
     *        data migration, calculation scheduling, and data migration operations
     * \param src, GM pointer for input data
     * \param dst, Gm pointer for outputting data
     * \param broadcastParam, dynamic parameters of broadcast, including tiling data, workspace, etc
     */
    template <typename... Args>
    __aicore__ inline void Run(Args&&... args)
    {
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][EleWise] Start to run template function.\n");
        constexpr std::size_t GM_ARGS_COUNT = INPUT_COUNT + OUTPUT_COUNT;
        GM_ADDR argsArr[INPUT_COUNT + OUTPUT_COUNT];
        InitHelper<0>(argsArr, ATVC::Forward<Args>(args)...);
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][EleWise] Template function execution completed.\n");
        pipeIn.Destroy();
    }

private:
    template <size_t N>
    __aicore__ inline void InitHelper(GM_ADDR tensorsGMArr[])
    {}

    template <size_t N, class T0, class... Ts>
    __aicore__ inline void InitHelper(GM_ADDR tensorsGMArr[], T0 t0, Ts... ts)
    {
        if constexpr (N < (INPUT_COUNT + OUTPUT_COUNT)) {
            tensorsGMArr[N] = t0;
            InitHelper<N + 1>(tensorsGMArr, ts...);
        } else if constexpr (N == (INPUT_COUNT + OUTPUT_COUNT)) {
            FillAddrs(tensorsGMArr);
            FillOffsets<Inputs>(inOffsets_);
            FillOffsets<Outputs>(outOffsets_);
            FillOffsets<Temps>(tempOffsets_);
            this->param_ = reinterpret_cast<ATVC::EleWiseParam*>(t0);
            // Check param
            ATVC::KernelUtils::PrintEleWiseParam<EleWiseParam>(param_);
            if (!ATVC::KernelUtils::CheckEleWiseParam<EleWiseParam>(param_)) {
                return;
            }
            uint32_t curBlockId = AscendC::GetBlockIdx();

            if (curBlockId < param_->tilingData.tailBlockCnt) {
                this->curCoreCnt_ = (param_->tilingData.numPerBlock + 1) * param_->tilingData.tiledCnt;
                this->curCoreStartCnt_ = (param_->tilingData.numPerBlock + 1) *
                                          curBlockId * param_->tilingData.tiledCnt;
            } else {
                this->curCoreCnt_ = param_->tilingData.numPerBlock * param_->tilingData.tiledCnt;
                this->curCoreStartCnt_ = (curBlockId * param_->tilingData.numPerBlock +
                                          param_->tilingData.tailBlockCnt) * param_->tilingData.tiledCnt;
            }
            if (curBlockId + 1 == param_->tilingData.blockNum) {
                this->curCoreCnt_ += param_->tilingData.tailElemCnt;
            }
            if (this->curCoreCnt_ == 0) {
                return;
            }
            Init();
            ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][EleWise] Initialize template execution completed.\n");
            Process(ATVC::Forward<Ts>(ts)...);
            ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][EleWise] Main process execution completed.\n");
        }
    }

private:
    __aicore__ inline bool DebugCheckTilingParam() {
        // tiledCnt、numPerBlock
        if (param_->tilingData.tiledCnt > param_->tilingData.numPerBlock) {
            return false;
        }
        //todo
        return true;
    }

    // Apply for localTensor and other resources to initialize the globalTensor for local core computing
    __aicore__ inline void Init()
    {
        // Each in/out/temp uses a pipe for management,
        // and each pipe manages multiple tensors with consecutive sub addresses
        if constexpr (INPUT_COUNT > 0) {
            GetTPipePtr()->InitBuffer(inQueue, param_->nBufferNum, param_->tilingData.tiledCnt * IN_TENSOR_SUM_BYTES);
        }
        if constexpr (OUTPUT_COUNT > 0) {
            GetTPipePtr()->InitBuffer(outQueue, param_->nBufferNum, param_->tilingData.tiledCnt * OUT_TENSOR_SUM_BYTES);
        }
        if constexpr(TEMP_COUNT > 0) {
            GetTPipePtr()->InitBuffer(tempQueue, param_->tilingData.tiledCnt * TEMP_TENSOR_SUM_BYTES);
        }
    }
    // Call CopyIn/CopyOut based on the tiling loop, as well as externally passed Compute calculations
    // If there is a tail block, process the CopyIn/Compute/CopyOut of the tail block
    template <typename... Args>
    __aicore__ inline void Process(Args&&... args)
    {
        typename TensorTuple<Inputs>::Type inTensors;
        typename TensorTuple<Outputs>::Type outTensors;
        typename TensorTuple<Temps>::Type tempTensors;

        InitInputTensors(inTensors, param_->tilingData.tiledCnt, ATVC::MakeIndexSequence<INPUT_COUNT>{});
        InitOutputTensors(outTensors, param_->tilingData.tiledCnt, ATVC::MakeIndexSequence<OUTPUT_COUNT>{});
        InitTempTensors(tempTensors, param_->tilingData.tiledCnt, ATVC::MakeIndexSequence<TEMP_COUNT>{});

        uint32_t repeat = curCoreCnt_ / param_->tilingData.tiledCnt;
        uint32_t tailCnt = curCoreCnt_ % param_->tilingData.tiledCnt;
        offsetCnt_ = 0;
        caclCnt_ = param_->tilingData.tiledCnt;
        // Loop processing of main block data
        for (uint32_t i = 0; i < repeat; i++) {
            CopyIn(inTensors, ATVC::MakeIndexSequence<INPUT_COUNT>{});
            Compute(inTensors, outTensors, tempTensors, ATVC::MakeIndexSequence<INPUT_COUNT>{},
                    ATVC::MakeIndexSequence<OUTPUT_COUNT>{}, ATVC::MakeIndexSequence<TEMP_COUNT>{},
                    ATVC::Forward<Args>(args)...);
            CopyOut(outTensors, ATVC::MakeIndexSequence<OUTPUT_COUNT>{});
            offsetCnt_ += caclCnt_;
        }

        // If there is a tail block, process the tail block
        if (tailCnt > 0) {
            caclCnt_ = tailCnt;
            CopyIn(inTensors, ATVC::MakeIndexSequence<INPUT_COUNT>{});
            Compute(inTensors, outTensors, tempTensors, ATVC::MakeIndexSequence<INPUT_COUNT>{},
                    ATVC::MakeIndexSequence<OUTPUT_COUNT>{}, ATVC::MakeIndexSequence<TEMP_COUNT>{},
                    ATVC::Forward<Args>(args)...);
            CopyOut(outTensors, ATVC::MakeIndexSequence<OUTPUT_COUNT>{});
        }
    }
    // Simulate the processing logic of a single Tensor: input the Tensor variable corresponding to type T
    template <typename T>
    __aicore__ inline void CopyInAllTensors(AscendC::LocalTensor<uint8_t>& inLocal, int32_t i, T& tensorInfo)
    {
        // The processing logic of a single Tensor
        auto inLocalI = inLocal[tensorInfo.localOffset].template ReinterpretCast<typename T::DataType>();

        using DataType = typename T::DataType;
        constexpr uint32_t typeAlignCnt = UB_ALIGN_32 / sizeof(DataType);
        uint32_t alignMainCnt = caclCnt_ / typeAlignCnt * typeAlignCnt;
        uint32_t alignTailCnt = caclCnt_ - alignMainCnt;
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][EleWise][CopyIn] Offset is %u, copy count is %u.\n",
                                  curCoreStartCnt_ + offsetCnt_, caclCnt_);
        if (alignMainCnt > 0) {
            AscendC::DataCopy(inLocalI, tensorInfo.gmTensor[curCoreStartCnt_ + offsetCnt_], alignMainCnt);
        }
        if (alignTailCnt > 0) {
            struct AscendC::DataCopyExtParams repeatParams = {1, (uint16_t)(alignTailCnt * sizeof(DataType)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DataType> padParams;
            AscendC::DataCopyPad(inLocalI[alignMainCnt],
                                 tensorInfo.gmTensor[curCoreStartCnt_ + offsetCnt_ + alignMainCnt], repeatParams,
                                 padParams);
        }
    }

    // Corresponding to the empty processing logic without Tensor
    __aicore__ inline void CopyInAllTensors(AscendC::LocalTensor<uint8_t>& inLocal, int32_t i) {}

    // The entry logic for processing all Tensors: recursively completing the processing of each Tensor
    template <typename T, typename... Ts>
    __aicore__ inline void CopyInAllTensors(AscendC::LocalTensor<uint8_t>& inLocal, int32_t i, T& first, Ts&... rest)
    {
        CopyInAllTensors(inLocal, i, first);
        CopyInAllTensors(inLocal, ++i, rest...);
    }

    // Copy all input tensors from gm to local
    template <typename InTuple, std::size_t... Is>
    __aicore__ inline void CopyIn(InTuple& inTensors, ATVC::IndexSequence<Is...>)
    {
        if constexpr (INPUT_COUNT == 0) {
            ATVC::Kernel::DebugPrintf("[ERROR]: [ATVC][EleWise] Input Count can not be 0!\n");
            return;
        }
        AscendC::LocalTensor<uint8_t> inLocal = inQueue.template AllocTensor<uint8_t>();
        // TypeListGetOffset is the offset of the current Tensor in bytes obtained from the TypeList
        // For example, TypeList<float, half, u8>, tensor_1 has an offset of 0, 
        // tensor_1 has an offset of sizeof (float), and tensor_2 has an offset of sizeof (float)+sizeof (half)
        CopyInAllTensors(inLocal, 0, ATVC::TupleElemGet<Is>(inTensors)...);
        inQueue.EnQue(inLocal);
    }

    // Simulate the processing logic of a single Tensor: input the Tensor variable corresponding to type T
    template <typename T>
    __aicore__ inline void CopyOutAllTensors(AscendC::LocalTensor<uint8_t>& outLocal, int32_t i, T& tensorInfo)
    {
        // The processing logic of a single Tensor
        auto outLocalI = outLocal[tensorInfo.localOffset].template ReinterpretCast<typename T::DataType>();
        using DataType = typename T::DataType;
        constexpr uint32_t TYPE_ALIGN_CNT = 32 / sizeof(DataType);
        uint32_t alignMainCnt = caclCnt_ / TYPE_ALIGN_CNT * TYPE_ALIGN_CNT;
        uint32_t alignTailCnt = caclCnt_ - alignMainCnt;
        ATVC::Kernel::DebugPrintf("[INFO]: [ATVC][EleWise][CopyOut] Offset is %u, copy count is %u.\n",
                                  curCoreStartCnt_ + offsetCnt_, caclCnt_);
        if (alignMainCnt > 0) {
            AscendC::DataCopy(tensorInfo.gmTensor[curCoreStartCnt_ + offsetCnt_], outLocalI, alignMainCnt);
        }
        if (alignTailCnt > 0) {
            struct AscendC::DataCopyParams repeatParams = {1, (uint16_t)(alignTailCnt * sizeof(DataType)), 0, 0};
            AscendC::DataCopyPad(tensorInfo.gmTensor[curCoreStartCnt_ + offsetCnt_ + alignMainCnt],
                                 outLocalI[alignMainCnt], repeatParams);
        }
    }

    // Corresponding to the empty processing logic without Tensor
    __aicore__ inline void CopyOutAllTensors(AscendC::LocalTensor<uint8_t>& outLocal, int32_t i) {
    }

    // The entry logic for processing all Tensors: recursively completing the processing of each Tensor
    template <typename T, typename... Ts>
    __aicore__ inline void
    CopyOutAllTensors(AscendC::LocalTensor<uint8_t> &outLocal, int32_t i,
                      T &first, Ts &...rest)
    {
      CopyOutAllTensors(outLocal, i, first);
      CopyOutAllTensors(outLocal, ++i, rest...);
    }
    // Copy all output tensors to gm
    template <typename OutTuple, std::size_t... Is>
    __aicore__ inline void CopyOut(OutTuple& outTensors, ATVC::IndexSequence<Is...>)
    {
      AscendC::LocalTensor<uint8_t> outLocal = outQueue.template DeQue<uint8_t>();
      CopyOutAllTensors(outLocal, 0, ATVC::TupleElemGet<Is>(outTensors)...);
      outQueue.FreeTensor(outLocal);
    }

    template<typename InTuple, typename OutTuple, typename TmpTuple,
        std::size_t... I1, std::size_t... I2,  std::size_t... I3, typename ...Args>
    __aicore__ inline void Compute(InTuple& inTensors, OutTuple& outTensors, TmpTuple& tempTensors,
        ATVC::IndexSequence<I1...>, ATVC::IndexSequence<I2...>, ATVC::IndexSequence<I3...>, Args&&... args)
    {
        AscendC::LocalTensor<uint8_t> inLocal = inQueue.template DeQue<uint8_t>();
        AscendC::LocalTensor<uint8_t> outLocal = outQueue.template AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint8_t> tempLocal;
        if constexpr(TEMP_COUNT > 0) {
            tempLocal = tempQueue.Get<uint8_t>();
        }
        compute_(TupleElemGetLocalTensor<I1>(inLocal, inTensors, this->caclCnt_)...,
            TupleElemGetLocalTensor<I2>(outLocal, outTensors, this->caclCnt_)...,
            TupleElemGetLocalTensor<I3>(tempLocal, tempTensors, this->caclCnt_)...,
            ATVC::Forward<Args>(args)...);
        inQueue.FreeTensor(inLocal);
        outQueue.template EnQue<uint8_t>(outLocal);
    }

    template <typename InTuple, std::size_t... Is>
    __aicore__ inline void InitInputTensors(InTuple& tuple, std::size_t cnt, ATVC::IndexSequence<Is...>)
    {
    // Initialize each Tensor
    [[maybe_unused]] int32_t dummy[] = {0, (InitInputTensor(ATVC::TupleElemGet<Is>(tuple), cnt, Is), 0)...};
    }

    template <typename OutTuple, std::size_t... Is>
    __aicore__ inline void InitOutputTensors(OutTuple& tuple, std::size_t cnt, ATVC::IndexSequence<Is...>)
    {
    [[maybe_unused]] int32_t dummy[] = {0, (InitOutputTensor(ATVC::TupleElemGet<Is>(tuple), cnt, Is), 0)...};
    }

    template <typename TmpTuple, std::size_t... Is>
    __aicore__ inline void InitTempTensors(TmpTuple& tuple, std::size_t cnt, ATVC::IndexSequence<Is...>)
    {
    [[maybe_unused]] int32_t dummy[] = {0, (InitTempTensor(ATVC::TupleElemGet<Is>(tuple), cnt, Is), 0)...};
    }

    template <typename T>
    __aicore__ inline TensorInfo<T>& InitInputTensor(TensorInfo<T>& tensor, std::size_t cnt, std::size_t index)
    {
    tensor.gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inGMAddrs_[index].GetPhyAddr(0)));
    tensor.localOffset = inOffsets_[index] * cnt;
    return tensor;
    }

    template <typename T>
    __aicore__ inline TensorInfo<T>& InitOutputTensor(TensorInfo<T>& tensor, std::size_t cnt, std::size_t index)
    {
    tensor.gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outGMAddrs_[index].GetPhyAddr(0)));
    tensor.localOffset = outOffsets_[index] * cnt;
    return tensor;
    }

    template <typename T>
    __aicore__ inline TensorInfo<T>& InitTempTensor(TensorInfo<T>& tensor, std::size_t cnt, std::size_t index)
    {
    tensor.localOffset = tempOffsets_[index] * cnt;
    return tensor;
    }

    // 填充 addr 到数组
    __aicore__ inline void FillAddrs(GM_ADDR argsArr[])
    {
        for (std::size_t i = 0; i < INPUT_COUNT; ++i) {
            inGMAddrs_[i].SetGlobalBuffer(argsArr[i]);
        }
        for (std::size_t i = 0; i < OUTPUT_COUNT; ++i) {
            outGMAddrs_[i].SetGlobalBuffer(argsArr[INPUT_COUNT + i]);
        }
    }

    // 填充 offset 到数组
    template <typename List, std::size_t... Is>
    __aicore__ inline constexpr void FillOffsetsImpl(std::size_t* offsets, ATVC::IndexSequence<Is...>)
    {
        ((offsets[Is] = ATVC::TypeListByteOffset<List, Is>::VALUE), ...);
    }

    template <typename List>
    __aicore__ inline constexpr void FillOffsets(std::size_t* offsets)
    {
        constexpr std::size_t count = ATVC::TypeListSize<List>::VALUE;
        FillOffsetsImpl<List>(offsets, ATVC::MakeIndexSequence<count>{});
    }

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempQueue;
    
    AscendC::GlobalTensor<uint8_t> inGMAddrs_[INPUT_COUNT];
    AscendC::GlobalTensor<uint8_t> outGMAddrs_[OUTPUT_COUNT];

    std::size_t inOffsets_[INPUT_COUNT];
    std::size_t outOffsets_[OUTPUT_COUNT];
    std::size_t tempOffsets_[TEMP_COUNT];

    // Calculated tiling data
    ATVC::EleWiseParam* param_;

    uint32_t curCoreCnt_;
    uint32_t curCoreStartCnt_;
    int32_t offsetCnt_;
    int32_t caclCnt_;

    // The calculation object passed in by user
    EleWiseCompute compute_;
    AscendC::TPipe pipeIn;
};
}
}
#endif