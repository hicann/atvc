/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "elewise/host/elewise_host.h"
#include "register/op_def_registry.h"

using AddOpTraitsFloat = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;
using AddOpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int32_t, int32_t>, ATVC::OpOutputs<int32_t>>;
   
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 声明运行态参数tiling
    ATVC::EleWiseParam *tiling = context->GetTilingData<ATVC::EleWiseParam>();
    uint32_t totleLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_FLOAT) {
        // AddOpTraitsFloat为ADD算子描述原型，根据算子输入输出个数和实际元素数量计算出Tiling数据后填入tiling中
        (void)ATVC::Host::CalcEleWiseTiling<AddOpTraitsFloat>(totleLength, *tiling);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_INT32) {
        (void)ATVC::Host::CalcEleWiseTiling<AddOpTraitsInt>(totleLength, *tiling);
    }
    // 设置tilingkey
    context->SetTilingKey(0);
    // 设置blockDim的大小
    context->SetBlockDim(tiling->tilingData.blockNum);
    // 设置Workspace的大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(AddCustom);
}  // namespace ops
