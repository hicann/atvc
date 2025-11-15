/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce/host/reduce_host.h"
#include "register/op_def_registry.h"

using ReduceOpTraitsFloat = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
using ReduceOpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int32_t>, ATVC::OpOutputs<int32_t>>;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ATVC::ReducePolicy policy = {0, 0, 0};
    auto inputShape0 = context->GetInputShape(0)->GetOriginShape();
    std::vector<int64_t> shapeIn;
    for (int32_t i = 0; i < inputShape0.GetDimNum(); i++) {
        shapeIn.push_back(inputShape0.GetDim(i));
    }
    // 获取dim值
    const gert::RuntimeAttrs *runtimeAttrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t> *attr0 = runtimeAttrs->GetListInt(0);
    const int64_t *arr = reinterpret_cast<const int64_t *>(attr0->GetData());    
    std::vector<int64_t> dim(arr, arr + attr0->GetSize());
    ATVC::ReduceParam *tiling = context->GetTilingData<ATVC::ReduceParam>();
    // default value is missed, need to restore
    tiling->nBufferNum = 2; // 2 : default value
    if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_FLOAT) {
         // ReduceOpTraitsFloat为Reduce算子描述原型，根据算子输入shape和dim计算出Tiling数据后填入tiling中
        (void)ATVC::Host::CalcReduceTiling<ReduceOpTraitsFloat>(shapeIn, dim, &policy, tiling);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_INT32) {
        (void)ATVC::Host::CalcReduceTiling<ReduceOpTraitsInt>(shapeIn, dim, &policy, tiling);
    }
    // 设置tiling的policyId为policy的id
    tiling->policyId = policy.getID();
    context->SetBlockDim(tiling->tilingData.coreNum);
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
class ReduceSumCustom : public OpDef {
public:
    explicit ReduceSumCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(REQUIRED).ListInt();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(ReduceSumCustom);
}  // namespace ops
