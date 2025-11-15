/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_ATVC_H
#define ATVC_ATVC_H

#include "common/atvc_opdef.h"
#include "common/atvc_op_check.h"
#include "common/const_def.h"

#include "elewise/common/elewise_common.h"
#include "elewise/host/elewise_host.h"

#include "reduce/common/reduce_common.h"
#include "reduce/host/reduce_host.h"

#include "broadcast/common/broadcast_common.h"
#include "broadcast/host/broadcast_host.h"

#include "kernel_operator.h"

#ifndef __NPU_HOST__
#include "common/kernel_utils.h"

#include "elewise/kernel/elewise_op_template.h"

#include "reduce/kernel/reduce_sum_compute.h"
#include "reduce/kernel/reduce_op_template.h"

#include "broadcast/kernel/broadcast_compute.h"
#include "broadcast/kernel/broadcast_op_template.h"
#endif

#endif