#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False
torch.ops.load_library('./libascendc_pytorch.so')


class TestAscendCOps(TestCase):

    def test_reduce_sum_ops_float(self):
        length = [8, 2048] 
        x = torch.rand(length, device='cpu', dtype=torch.float32) 
        npuout = torch.ops.ascendc_ops.sum(x.npu(), (0,))
        cpuout = torch.sum(x, (0,))
        self.assertRtolEqual(npuout.reshape(cpuout.shape), cpuout) 

    def test_reduce_sum_ops_int(self):
        length = [8, 2048] 
        x = torch.randint(-10, 10, length, device='cpu', dtype=torch.int32)
        npuout = torch.ops.ascendc_ops.sum(x.npu(), (0,))
        cpuout = torch.sum(x, (0,), dtype=torch.int32)
        self.assertRtolEqual(npuout.reshape(cpuout.shape), cpuout)

if __name__ == '__main__':
    run_tests()