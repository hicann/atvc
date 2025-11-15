/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_POOL_OP_TEMPLATE_H
#define ATVC_POOL_OP_TEMPLATE_H

#include <type_traits>
#include "kernel_operator.h"

#include "common/tensor_info.h"
#include "common/type_list.h"
#include "common/index_seq.h"
#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "common/kernel_check_debug.h"
#include "common/ops_utils_device.h"
#include "pool/common/pool_common.h"
#include "pool/kernel/pool_utils/pool_util.h"


namespace ATVC {
namespace Kernel {
struct TileOffset {
    uint32_t left;
    uint32_t up;
};

struct PoolTileScope {
    Layout2Dim layout;
    TileOffset offset;
    PoolTilePadding tilePadding;
};

template <class PoolCompute, const ATVC::Layout2Dim &totalLayout>
class PoolOpTemplate {
    using PoolOpTraits = typename GetFunctionTraits<PoolCompute>::ComputeTraits;
    using Inputs = typename PoolOpTraits::In::types;
    using Outputs = typename PoolOpTraits::Out::types;
    using Temps = typename PoolOpTraits::Temp::types;
    static constexpr Layout2Dim TILE_LAYOUT = PoolCompute::TILE_LAYOUT;         // Layout of tile block
    static constexpr PoolTilePadding TILE_PADDING = PoolCompute::TILE_PADDING;  // padding value of tile block
    static constexpr size_t INPUT_COUNT = ATVC::TypeListSize<Inputs>::VALUE;
    static constexpr size_t OUTPUT_COUNT = ATVC::TypeListSize<Outputs>::VALUE;
    static constexpr size_t TEMP_COUNT = ATVC::TypeListSize<Temps>::VALUE;

    static constexpr size_t IN_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Inputs, SizeValue<0>, SumSizes>::Type::VALUE;
    static constexpr size_t OUT_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Outputs, SizeValue<0>, SumSizes>::Type::VALUE;
    static constexpr size_t TEMP_TENSOR_SUM_BYTES = ATVC::TypeListReduce<Temps, SizeValue<0>, SumSizes>::Type::VALUE;

public:
    __aicore__ inline PoolOpTemplate()
    {}
    template <typename... Args>
    __aicore__ inline void Run(Args &&...args)
    {
        ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] Start to run template function.\n");
        constexpr std::size_t GM_ARGS_COUNT = INPUT_COUNT + OUTPUT_COUNT;
        GM_ADDR argsArr[INPUT_COUNT + OUTPUT_COUNT];
        InitHelper<0>(argsArr, ATVC::Forward<Args>(args)...);
        ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] End to run template function.\n");
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
            this->param_ = reinterpret_cast<ATVC::PoolParam *>(t0);
            // Check param
            ATVC::KernelUtils::PrintPoolParam<PoolParam>(param_);
            if (!ATVC::KernelUtils::CheckPoolParam<PoolParam>(param_)) {
                return;
            }
            CalcCurCoreStart();
            Init();
            ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] End to initialize template.\n");
            Process(ATVC::Forward<Ts>(ts)...);
            ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool] End main process.\n");
        }
    }

private:
    __aicore__ inline void CalcCurCoreStart()
    {
        uint32_t curBlockId = AscendC::GetBlockIdx();
        bool isCurCoreTail = curBlockId < param_->tilingData.tailBlockCnt;
        curCoreTileCnt_ = param_->tilingData.numPerBlock + (isCurCoreTail ? 1 : 0);
        uint32_t cntW = OpsUtils::CeilDiv(totalLayout.width, TILE_LAYOUT.width);
        if (cntW == 0) {
            cntW =1;
            ATVC::Kernel::DebugPrintf(
            "[INFO]:[ATVC][Error] Invalid Layout, may cause err! totalLayout.width = %u TILE_LAYOUT.width = %u \n",
            totalLayout.width,
            TILE_LAYOUT.width);
        }
        uint32_t curCoreStartCnt = param_->tilingData.numPerBlock * curBlockId;
        if (isCurCoreTail) {
            curCoreStartCnt += curBlockId;
        } else {
            curCoreStartCnt += param_->tilingData.tailBlockCnt;
        }

        curOffset_.up = (curCoreStartCnt / cntW) * TILE_LAYOUT.height;
        curOffset_.left = (curCoreStartCnt % cntW) * TILE_LAYOUT.width;
        basicTensorCnt_ = (TILE_LAYOUT.width + TILE_PADDING.left + TILE_PADDING.right) *
                          (TILE_LAYOUT.height + TILE_PADDING.up + TILE_PADDING.down);
        ATVC::Kernel::DebugPrintf(
            "[INFO]:[ATVC][Pool] curCoreStartCnt = %u curOffset_.up: %u curOffset_.left: %u basicTensorCnt_ = %u\n",
            curCoreStartCnt,
            curOffset_.up,
            curOffset_.left,
            basicTensorCnt_);
    }

    __aicore__ inline void UpdateCurTileOffset()
    {
        curOffset_.left += TILE_LAYOUT.width;
        if (curOffset_.left >= totalLayout.width) {
            curOffset_.left = 0;
            curOffset_.up += TILE_LAYOUT.height;
        }
    }

    __aicore__ inline void CalcCurTileScope()
    {
        // 1 data offset for cur tile scope
        curTileScope_.offset.left = curOffset_.left;  // need to be 32B byte aligned
        curTileScope_.offset.up = curOffset_.up;
        // 1.1 layout for cur tile scope
        curTileScope_.layout.width = curTileScope_.offset.left + TILE_LAYOUT.width > totalLayout.width
                                         ? totalLayout.width - curTileScope_.offset.left
                                         : TILE_LAYOUT.width;
        curTileScope_.layout.height = curTileScope_.offset.up + TILE_LAYOUT.height > totalLayout.height
                                          ? totalLayout.height - curTileScope_.offset.up
                                          : TILE_LAYOUT.height;

        // 1.2 padding for cur tile scope
        curTileScope_.tilePadding.up =
            curTileScope_.offset.up >= TILE_PADDING.up ? TILE_PADDING.up : curTileScope_.offset.up;
        curTileScope_.tilePadding.left =
            curTileScope_.offset.left >= TILE_PADDING.left ? TILE_PADDING.left : curTileScope_.offset.left;

        uint32_t totalUsed = curTileScope_.offset.left + curTileScope_.layout.width;
        curTileScope_.tilePadding.right =
            totalUsed + TILE_PADDING.right > totalLayout.width ? (totalLayout.width - totalUsed) : TILE_PADDING.right;
        totalUsed = curTileScope_.offset.up + curTileScope_.layout.height;
        curTileScope_.tilePadding.down =
            totalUsed + TILE_PADDING.down > totalLayout.height ? (totalLayout.height - totalUsed) : TILE_PADDING.down;

        ATVC::Kernel::DebugPrintf(
            "[INFO]:[ATVC][Pool]Curent offset.up: %u .left: %u  padding: up: %u left: %u right: %u down: %u\n",
            curTileScope_.offset.up,
            curTileScope_.offset.left,
            curTileScope_.tilePadding.up,
            curTileScope_.tilePadding.left,
            curTileScope_.tilePadding.right,
            curTileScope_.tilePadding.down);
    }
    // Init LocalTensor Queue
    __aicore__ inline void Init()
    {
        if constexpr (INPUT_COUNT > 0) {
            GetTPipePtr()->InitBuffer(inQueue, param_->nBufferNum, basicTensorCnt_ * channel_ * IN_TENSOR_SUM_BYTES);
        }
        if constexpr (OUTPUT_COUNT > 0) {
            GetTPipePtr()->InitBuffer(outQueue, param_->nBufferNum, basicTensorCnt_ * OUT_TENSOR_SUM_BYTES);
        }
        if constexpr (TEMP_COUNT > 0) {
            GetTPipePtr()->InitBuffer(tempQueue, basicTensorCnt_ * TEMP_TENSOR_SUM_BYTES);
        }
    }
    // Call CopyIn/CopyOut based on the tiling loop, as well as externally passed Compute calculations
    // If there is a tail block, process the CopyIn/Compute/CopyOut of the tail block
    template <typename... Args>
    __aicore__ inline void Process(Args &&...args)
    {
        typename TensorTuple<Inputs>::Type inTensors;
        typename TensorTuple<Outputs>::Type outTensors;
        typename TensorTuple<Temps>::Type tempTensors;

        InitInputTensors(inTensors, basicTensorCnt_ * channel_, ATVC::MakeIndexSequence<INPUT_COUNT>{});
        InitOutputTensors(outTensors, basicTensorCnt_, ATVC::MakeIndexSequence<OUTPUT_COUNT>{});
        InitTempTensors(tempTensors, basicTensorCnt_, ATVC::MakeIndexSequence<TEMP_COUNT>{});
        // Loop processing of block data
        for (uint32_t i = 0; i < curCoreTileCnt_; i++) {
            AscendC::PipeBarrier<PIPE_ALL>();
            CalcCurTileScope();
            CopyIn(inTensors, ATVC::MakeIndexSequence<INPUT_COUNT>{});
            AscendC::PipeBarrier<PIPE_ALL>();
            Compute(inTensors,
                outTensors,
                tempTensors,
                ATVC::MakeIndexSequence<INPUT_COUNT>{},
                ATVC::MakeIndexSequence<OUTPUT_COUNT>{},
                ATVC::MakeIndexSequence<TEMP_COUNT>{},
                ATVC::Forward<Args>(args)...);
            AscendC::PipeBarrier<PIPE_ALL>();
            CopyOut(outTensors, ATVC::MakeIndexSequence<OUTPUT_COUNT>{});
            AscendC::PipeBarrier<PIPE_ALL>();
            UpdateCurTileOffset();
        }
    }

    // Copy data from GM to UB for all input params.
    template <typename T>
    __aicore__ inline void CopyInAllTensors(AscendC::LocalTensor<uint8_t> &inLocal, int32_t i, T &tensorInfo)
    {
        // Processing for per input LocalTensor
        auto inLocalI = inLocal[tensorInfo.localOffset].template ReinterpretCast<typename T::DataType>();

        using DataType = typename T::DataType;
        AscendC::Duplicate<DataType>(inLocalI, 0, inLocalI.GetSize());
        constexpr uint32_t typeAlignCnt = UB_ALIGN_32 / sizeof(DataType);

        // Calc data offset
        uint32_t uBTensorW = (TILE_LAYOUT.width + TILE_PADDING.left + TILE_PADDING.right);
        uint32_t copyGmOffset = ((curTileScope_.offset.up - curTileScope_.tilePadding.up) * totalLayout.width +
                                    (curTileScope_.offset.left - curTileScope_.tilePadding.left)) *
                                channel_;

        // offset for gm to UB
        uint32_t ubCopyOffset = (TILE_PADDING.up - curTileScope_.tilePadding.up) * uBTensorW;
        ubCopyOffset = ubCopyOffset * channel_;

        // data copy params
        uint32_t copyBlockW =
            curTileScope_.tilePadding.left + curTileScope_.tilePadding.right + curTileScope_.layout.width;
        uint16_t copyBlockH =
            curTileScope_.tilePadding.up + curTileScope_.tilePadding.down + curTileScope_.layout.height;
        uint32_t srcStride = (totalLayout.width - copyBlockW) * sizeof(DataType) * channel_;  // srcStride: B
        uint32_t dstStride = 0;                                                               // dstStride: 32B
        // width_ * channel_ * sizeof(DataType) and ubBlockLen need to be 32B aligned
        struct AscendC::DataCopyExtParams repeatParams = {copyBlockH,
            static_cast<uint16_t>(copyBlockW * channel_ * sizeof(DataType)),  // uint is B
            srcStride,
            dstStride,
            0};
        AscendC::DataCopyPadExtParams<DataType> padParams{true,
            static_cast<uint8_t>(TILE_PADDING.left - curTileScope_.tilePadding.left),
            static_cast<uint8_t>(TILE_PADDING.right - curTileScope_.tilePadding.right),
            0};
        AscendC::DataCopyPad(inLocalI[ubCopyOffset], tensorInfo.gmTensor[copyGmOffset], repeatParams, padParams);
        ATVC::Kernel::DebugPrintf("[INFO]:[ATVC][Pool][CopyIn]ubCopyOffset: %u, copyGmOffset: %u, copyBlockW: %u, "
                                  "srcStride: %u dstStride: %u curTileScope_:tile left %u right %u width %u\n",
            ubCopyOffset,
            copyGmOffset,
            copyBlockW,
            srcStride,
            dstStride,
            curTileScope_.tilePadding.left,
            curTileScope_.tilePadding.right,
            curTileScope_.layout.width);
    }

    __aicore__ inline void CopyInAllTensors(AscendC::LocalTensor<uint8_t> &inLocal, int32_t i)
    {}

    // The entry logic for processing all Tensors: recursively completing the processing of each Tensor
    template <typename T, typename... Ts>
    __aicore__ inline void CopyInAllTensors(AscendC::LocalTensor<uint8_t> &inLocal, int32_t i, T &first, Ts &...rest)
    {
        CopyInAllTensors(inLocal, i, first);
        CopyInAllTensors(inLocal, ++i, rest...);
    }

    // Copy all input tensors from gm to local
    template <typename InTuple, std::size_t... Is>
    __aicore__ inline void CopyIn(InTuple &inTensors, ATVC::IndexSequence<Is...>)
    {
        if constexpr (INPUT_COUNT == 0) {
            ATVC::Kernel::DebugPrintf("[ERROR]:[ATVC][Pool] Input Count can not be 0!\n");
            return;
        }
        AscendC::LocalTensor<uint8_t> inLocal = inQueue.template AllocTensor<uint8_t>();
        CopyInAllTensors(inLocal, 0, ATVC::TupleElemGet<Is>(inTensors)...);
        inQueue.EnQue(inLocal);
    }

    template <typename T>
    __aicore__ inline void CopyOutAllTensors(AscendC::LocalTensor<uint8_t> &outLocal, int32_t i, T &tensorInfo)
    {
        // Calc offset
        uint32_t uBTensorW = (TILE_LAYOUT.width + TILE_PADDING.left + TILE_PADDING.right);
        uint32_t copyGmOffset = curTileScope_.offset.up * totalLayout.width + curTileScope_.offset.left;
        uint32_t ubCopyOffset =
            TILE_PADDING.up * uBTensorW + TILE_PADDING.left;  // UB tensor layout depend to TILE_PADDING + TILE_LAYOUT
        // Process for singer tensor
        auto outLocalI = outLocal[tensorInfo.localOffset].template ReinterpretCast<typename T::DataType>();

        using DataType = typename T::DataType;
        constexpr uint32_t typeAlignCnt = UB_ALIGN_32 / sizeof(DataType);
        uint16_t srcStride = (uBTensorW - curTileScope_.layout.width) / typeAlignCnt;  // srcStride: 32B aligned
        uint16_t dstStride = (totalLayout.width - curTileScope_.layout.width) * sizeof(DataType);
        // curTileH need tobe in [0, 4096], otherwise, need multi data copy.
        struct AscendC::DataCopyParams repeatParams = {static_cast<uint16_t>(curTileScope_.layout.height),
            static_cast<uint16_t>(curTileScope_.layout.width * sizeof(DataType)),
            srcStride,
            dstStride};
        AscendC::DataCopyPad(tensorInfo.gmTensor[copyGmOffset], outLocalI[ubCopyOffset], repeatParams);
        ATVC::Kernel::DebugPrintf(
            "[INFO]:[ATVC][Pool][CopyOut]ubCopyOffset: %u, copyGmOffset: %u, srcStride: %u dstStride: %u\n",
            ubCopyOffset,
            copyGmOffset,
            srcStride,
            dstStride);
    }

    __aicore__ inline void CopyOutAllTensors(AscendC::LocalTensor<uint8_t> &outLocal, int32_t i)
    {}

    template <typename T, typename... Ts>
    __aicore__ inline void CopyOutAllTensors(AscendC::LocalTensor<uint8_t> &outLocal, int32_t i, T &first, Ts &...rest)
    {
        CopyOutAllTensors(outLocal, i, first);
        CopyOutAllTensors(outLocal, ++i, rest...);
    }

    template <typename OutTuple, std::size_t... Is>
    __aicore__ inline void CopyOut(OutTuple &outTensors, ATVC::IndexSequence<Is...>)
    {
        AscendC::LocalTensor<uint8_t> outLocal = outQueue.template DeQue<uint8_t>();
        CopyOutAllTensors(outLocal, 0, ATVC::TupleElemGet<Is>(outTensors)...);
        outQueue.FreeTensor(outLocal);
    }

    template <typename InTuple, typename OutTuple, typename TmpTuple, std::size_t... I1, std::size_t... I2,
        std::size_t... I3, typename... Args>
    __aicore__ inline void Compute(InTuple &inTensors, OutTuple &outTensors, TmpTuple &tempTensors,
        ATVC::IndexSequence<I1...>, ATVC::IndexSequence<I2...>, ATVC::IndexSequence<I3...>, Args &&...args)
    {
        AscendC::LocalTensor<uint8_t> inLocal = inQueue.template DeQue<uint8_t>();
        AscendC::LocalTensor<uint8_t> outLocal = outQueue.template AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint8_t> tempLocal;
        if constexpr (TEMP_COUNT > 0) {
            tempLocal = tempQueue.Get<uint8_t>();
        }
        compute_(TupleElemGetLocalTensor<I1>(inLocal, inTensors, basicTensorCnt_ * channel_)...,
            TupleElemGetLocalTensor<I2>(outLocal, outTensors, basicTensorCnt_)...,
            TupleElemGetLocalTensor<I3>(tempLocal, tempTensors, basicTensorCnt_)...,
            ATVC::Forward<Args>(args)...);
        inQueue.FreeTensor(inLocal);
        outQueue.template EnQue<uint8_t>(outLocal);
    }

    template <typename InTuple, std::size_t... Is>
    __aicore__ inline void InitInputTensors(InTuple &tuple, std::size_t cnt, ATVC::IndexSequence<Is...>)
    {
        [[maybe_unused]] int32_t dummy[] = {0, (InitInputTensor(ATVC::TupleElemGet<Is>(tuple), cnt, Is), 0)...};
    }

    template <typename OutTuple, std::size_t... Is>
    __aicore__ inline void InitOutputTensors(OutTuple &tuple, std::size_t cnt, ATVC::IndexSequence<Is...>)
    {
        [[maybe_unused]] int32_t dummy[] = {0, (InitOutputTensor(ATVC::TupleElemGet<Is>(tuple), cnt, Is), 0)...};
    }

    template <typename TmpTuple, std::size_t... Is>
    __aicore__ inline void InitTempTensors(TmpTuple &tuple, std::size_t cnt, ATVC::IndexSequence<Is...>)
    {
        [[maybe_unused]] int32_t dummy[] = {0, (InitTempTensor(ATVC::TupleElemGet<Is>(tuple), cnt, Is), 0)...};
    }

    template <typename T>
    __aicore__ inline TensorInfo<T> &InitInputTensor(TensorInfo<T> &tensor, std::size_t cnt, std::size_t index)
    {
        tensor.gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inGMAddrs_[index].GetPhyAddr(0)));
        tensor.localOffset = inOffsets_[index] * cnt;
        return tensor;
    }

    template <typename T>
    __aicore__ inline TensorInfo<T> &InitOutputTensor(TensorInfo<T> &tensor, std::size_t cnt, std::size_t index)
    {
        tensor.gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outGMAddrs_[index].GetPhyAddr(0)));
        tensor.localOffset = outOffsets_[index] * cnt;
        return tensor;
    }

    template <typename T>
    __aicore__ inline TensorInfo<T> &InitTempTensor(TensorInfo<T> &tensor, std::size_t cnt, std::size_t index)
    {
        tensor.localOffset = tempOffsets_[index] * cnt;
        return tensor;
    }

    __aicore__ inline void FillAddrs(GM_ADDR argsArr[])
    {
        for (std::size_t i = 0; i < INPUT_COUNT; ++i) {
            inGMAddrs_[i].SetGlobalBuffer(argsArr[i]);
        }
        for (std::size_t i = 0; i < OUTPUT_COUNT; ++i) {
            outGMAddrs_[i].SetGlobalBuffer(argsArr[INPUT_COUNT + i]);
        }
    }

    template <typename List, std::size_t... Is>
    __aicore__ inline constexpr void FillOffsetsImpl(std::size_t *offsets, ATVC::IndexSequence<Is...>)
    {
        ((offsets[Is] = ATVC::TypeListByteOffset<List, Is>::VALUE), ...);
    }

    template <typename List>
    __aicore__ inline constexpr void FillOffsets(std::size_t *offsets)
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

    ATVC::PoolParam *param_;

    uint32_t curCoreTileCnt_;
    uint32_t channel_ = 1;  // C Dim
    TileOffset curOffset_{0, 0};
    PoolTileScope curTileScope_;
    uint32_t basicTensorCnt_;
    AscendC::TPipe pipeIn;
    PoolCompute compute_;
};
}  // namespace Kernel
}  // namespace ATVC
#endif