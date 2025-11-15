/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_REDUCE_COMMON_H
#define ATVC_REDUCE_COMMON_H

#include "patterns.h"
namespace ATVC {
enum ShapeDim {
    DIM_0,
    DIM_1,
    DIM_2,
    DIM_3,
    DIM_4,
    DIM_5,
    DIM_6,
    DIM_7,
    DIM_8,
    DIM_9,
    DIM_REDUCE,     // Reduce axis
    DIM_BROADCAST   // Broadcast axis
};

namespace AR_PATTERN {
    static constexpr uint32_t A = 100;
    static constexpr uint32_t AR = 11;
    static constexpr uint32_t ARA = 20;
    static constexpr uint32_t ARAR = 31;
    static constexpr uint32_t ARARA = 40;
};

namespace BASIC_CNT {
    static constexpr uint32_t BASIC_128 = 128;
    static constexpr uint32_t BASIC_256 = 256;
    static constexpr uint32_t BASIC_512 = 512;
    static constexpr uint32_t BASIC_1024 = 1024;
    static constexpr uint32_t BASIC_2048 = 2048;
    static constexpr uint32_t BASIC_4096 = 4096;
};

namespace AR_COUNT {
    static constexpr uint32_t A0R1 = 1;
    static constexpr uint32_t A0R2 = 2;
    static constexpr uint32_t A1R0 = 10;
    static constexpr uint32_t A2R0 = 20;
    static constexpr uint32_t A3R0 = 30;
    static constexpr uint32_t A4R0 = 40;
    static constexpr uint32_t A5R0 = 50;
    static constexpr uint32_t A1R2 = 12;
    static constexpr uint32_t A1R3 = 13;
    static constexpr uint32_t A1R4 = 14;
    static constexpr uint32_t A1R5 = 15;
    static constexpr uint32_t A2R3 = 23;
    static constexpr uint32_t A2R4 = 24;
    static constexpr uint32_t A2R5 = 25;
    static constexpr uint32_t A2R6 = 26;
    static constexpr uint32_t A3R4 = 34;
    static constexpr uint32_t A3R5 = 35;
    static constexpr uint32_t A3R6 = 36;
    static constexpr uint32_t A3R7 = 37;
    static constexpr uint32_t A4R5 = 45;
    static constexpr uint32_t A4R6 = 46;
    static constexpr uint32_t A4R7 = 47;
    static constexpr uint32_t A4R8 = 48;
    static constexpr uint32_t A5R6 = 56;
    static constexpr uint32_t A5R7 = 57;
    static constexpr uint32_t A5R8 = 58;
    static constexpr uint32_t A5R9 = 59;
};

constexpr int32_t poly1 = 1000000;
constexpr int32_t poly2 = 1000;

struct ReducePolicy {
public:
    int32_t patternID = -1;
    int32_t loopARCount = -1;
    int32_t loopInnerARCount = -1;
    int32_t ID = -1;
    constexpr ReducePolicy(int32_t patternID, int32_t loopARCount, int32_t loopInnerARCount)
        : patternID(patternID), loopARCount(loopARCount), loopInnerARCount(loopInnerARCount), 
            ID(poly1 * patternID + poly2 * loopARCount + loopInnerARCount)
    {}
    bool operator==(const ReducePolicy& rhs) const
    {
        return this->patternID == rhs.patternID && this->loopARCount == rhs.loopARCount &&\
        this->loopInnerARCount == rhs.loopInnerARCount;
    }
    int32_t getID() {
        return poly1 * this->patternID + poly2 * this->loopARCount + this->loopInnerARCount;
    }
};

static constexpr ReducePolicy REDUCE_POLICY0 { AR_PATTERN::A, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY1 { AR_PATTERN::AR, AR_COUNT::A0R1, 10 };
static constexpr ReducePolicy REDUCE_POLICY2 { AR_PATTERN::AR, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY3 { AR_PATTERN::AR, AR_COUNT::A1R0, 1 };
static constexpr ReducePolicy REDUCE_POLICY4 { AR_PATTERN::AR, AR_COUNT::A1R2, 0 };
static constexpr ReducePolicy REDUCE_POLICY5 { AR_PATTERN::ARA, AR_COUNT::A0R1, 10 };
static constexpr ReducePolicy REDUCE_POLICY6 { AR_PATTERN::ARA, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY7 { AR_PATTERN::ARA, AR_COUNT::A1R0, 1 };
static constexpr ReducePolicy REDUCE_POLICY8 { AR_PATTERN::ARA, AR_COUNT::A1R2, 0 };
static constexpr ReducePolicy REDUCE_POLICY9 { AR_PATTERN::ARA, AR_COUNT::A2R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY10 { AR_PATTERN::ARA, AR_COUNT::A2R0, 1 };
static constexpr ReducePolicy REDUCE_POLICY11 { AR_PATTERN::ARA, AR_COUNT::A2R3, 0 };
static constexpr ReducePolicy REDUCE_POLICY12 { AR_PATTERN::ARAR, AR_COUNT::A0R2, 10 };
static constexpr ReducePolicy REDUCE_POLICY13 { AR_PATTERN::ARAR, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY14 { AR_PATTERN::ARAR, AR_COUNT::A1R0, 2 };
static constexpr ReducePolicy REDUCE_POLICY15 { AR_PATTERN::ARAR, AR_COUNT::A2R0, 1 };
static constexpr ReducePolicy REDUCE_POLICY16 { AR_PATTERN::ARAR, AR_COUNT::A2R0, 2 };
static constexpr ReducePolicy REDUCE_POLICY17 { AR_PATTERN::ARAR, AR_COUNT::A2R4, 0 };
static constexpr ReducePolicy REDUCE_POLICY18 { AR_PATTERN::ARARA, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY19 { AR_PATTERN::ARARA, AR_COUNT::A2R0, 2 };
static constexpr ReducePolicy REDUCE_POLICY20 { AR_PATTERN::ARARA, AR_COUNT::A3R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY21 { AR_PATTERN::ARARA, AR_COUNT::A3R0, 2 };
static constexpr ReducePolicy REDUCE_POLICY22 { AR_PATTERN::ARARA, AR_COUNT::A2R0, 0 };

struct ReduceTilingData {
    uint64_t factorACntPerCore; // The actual dimensions of non Reduce axes that do not participate
                                // in computation on each core
    uint64_t factorATotalCnt;   // The total dimension of non Reduce axes that do not participate in the calculation
    uint64_t ubFactorA;         // The amount of data on non Reduce axes within a single UB
    uint64_t factorRCntPerCore; // The actual dimension of the Reduce axis involved in computation on each core
    uint64_t factorRTotalCnt;   // The total dimension of the Reduce axis involved in the calculation
    uint64_t ubFactorR;         // Reduce axis dimension involved in calculation within a single UB
    uint64_t groupR;            // The tangent axis is the R axis, and the relative data amount of R
                                // outside the tangent point on this axis
    uint64_t outSize;           // The total amount of AR data outside the cutting axis
    uint64_t basicBlock;        // The basic data block size
    int32_t coreNum;            // The number of running cores
    float meanVar;              // Reserved
    uint64_t shape[MAX_DIM];        //  Shape info
    uint64_t stride[MAX_DIM];       //  Input data transfer step size
    uint64_t dstStride[MAX_DIM];    // Output data transfer step size
};

struct ReduceParam {
    uint64_t workspaceAddr;         // The address of the requested space on the device
    uint32_t workspaceSize = 0;     // The size of the requested space
    ReduceTilingData tilingData;    // tilingData
    int32_t nBufferNum = 2;         // The number of Tensors in each Queue
    int32_t policyId = -1;
};

struct ReduceSchLoopInfo {
    int32_t patternID;
    int32_t reduceDichotomy;
    int32_t loopACount;
    int32_t loopAAxis[ATVC::ReducePattern::MAX_LOOP_DIM];
    int32_t loopRCount;
    int32_t loopRAxis[ATVC::MAX_DIM];

    int32_t loopInnerACount;
    int32_t loopInnerAAxis[ATVC::MAX_DIM];
    int32_t loopInnerRCount;
    int32_t loopInnerRAxis[ATVC::MAX_DIM];
    int32_t innerPatternID;
};
};

#endif // ATVC_REDUCE_COMMON_H